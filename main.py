import os
import math
import time
# import wandb

import torch
import shutil
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from monai.data import DataLoader, decollate_batch
from monai.utils import set_determinism
from monai.data.utils import list_data_collate
from monai.metrics import MSEMetric, ROCAUCMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
)

import config
import misc
import load_data
import process_data
import data_preproc.data_preproc_config as data_preproc_config

from data_preproc.data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from data_preproc.checks.check_data_preproc import plot_multiple_arrs
from models.logistic_regression import run_logistic_regression


#torch.multiprocessing.set_sharing_strategy('file_system')


def initialize(torch_version, device, create_folders=True):
    """
    Set up experiment and save the experiment path to be able to save results.

    Args:
        torch_version:
        device:
        create_folders:

    Returns:

    """
    if create_folders:
        # Create experiment folder and subfolders if they do not exist yet
        exp_dir = config.exp_dir
        # Subfolders
        exp_src_dir = config.exp_src_dir
        exp_models_dir = config.exp_models_dir
        exp_optimizers_dir = config.exp_optimizers_dir
        exp_data_preproc_dir = config.exp_data_preproc_dir
        exp_figures_dir = config.exp_figures_dir

        for p in [exp_dir, exp_src_dir, exp_models_dir, exp_optimizers_dir, exp_data_preproc_dir, exp_figures_dir]:
            create_folder_if_not_exists(p)

        # Fetch folders and files that should be copied to exp folder
        root_path = config.root_path
        models_dir = config.models_dir
        optimizers_dir = config.optimizers_dir
        data_preproc_dir = config.data_preproc_dir
        src_files = [x for x in os.listdir(root_path) if
                     (x.endswith('.py') or x == data_preproc_config.filename_exclude_patients_csv)]

        # Copy src files to exp folder
        for f in src_files:
            copy_file(src=os.path.join(root_path, f), dst=os.path.join(exp_src_dir, f))
        # Copy folders to exp folder
        copy_folder(src=models_dir, dst=exp_models_dir)
        copy_folder(src=optimizers_dir, dst=exp_optimizers_dir)
        copy_folder(src=data_preproc_dir, dst=exp_data_preproc_dir)

        # Logger output filename
        output_filename = os.path.join(exp_dir, 'log.txt')
    else:
        output_filename = None

    # Initialize logger
    logger = Logger(output_filename=output_filename)

    # Print variables
    logger.my_print('Device: {}.'.format(device))
    logger.my_print('Torch version: {}.'.format(torch_version))
    logger.my_print('Torch.backends.cudnn.benchmark: {}.'.format(torch.backends.cudnn.benchmark))

    return logger


def weights_init(m, weight_init_name, kaiming_a, kaiming_mode, kaiming_nonlinearity, gain, logger):
    """
    Custom weights initialization.

    The weights_init function takes an initialized model as input and re-initializes all convolutional,
    batch normalization and instance normalization layers.

    Source: https://pytorch.org/docs/stable/nn.init.html
    Source: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/conv.py#L49
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/linear.py#L83
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/batchnorm.py#L51
    Source: https://discuss.pytorch.org/t/initialization-and-batch-normalization/30655/2

    Args:
        m:
        weight_init_name:
        kaiming_a:
        kaiming_mode:
        kaiming_nonlinearity:
        gain:
        logger:

    Returns:

    """
    # Initialize variables
    classname = m.__class__.__name__

    # Initialize output layer, for which a sigmoid-function will be preceded.
    if 'Output' in classname:
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)
        logger.my_print('Weights init of output layer: Xavier uniform.')

    elif ('Conv' in classname) or ('Linear' in classname):
        if weight_init_name == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'uniform':
            torch.nn.init.uniform_(m.weight)
        elif weight_init_name == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        elif weight_init_name == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'normal':
            torch.nn.init.normal_(m.weight)
        elif weight_init_name == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
        elif weight_init_name == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight, gain=gain)
        else:
            raise ValueError('Invalid weight_init_name: {}.'.format(weight_init_name))

        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)


def validate(model, dataloader, mode, logger, mean, std, save_outputs=True,):
    """
    Validate the model.

    Args:
        model:
        dataloader:
        mode (str): for printing, either 'internal validation' or 'test'
        logger:

    Returns:

    """
    # Initialize variable
    model.eval()
    mode = mode.capitalize()
    loss_value = 0
    num_iterations = len(dataloader)
    patient_ids = list()

    with torch.no_grad():
        y_pred = torch.as_tensor([], dtype=torch.float32, device=device)
        y = torch.as_tensor([], dtype=torch.int8, device=device)
        for data in dataloader:
            # Load data
            # ct, rtdose, segmentation_map, features, labels = (
            #     data['ct'].to(device),
            #     data['rtdose'].to(device),
            #     data['segmentation_map'].to(device),
            #     data['features'].to(device),
            #     data['label'].to(device),
            # )
            # inputs = torch.concat([ct, rtdose, segmentation_map], axis=1)  # torch.Size([1, 3, 100, 100, 100])
            inputs, features, labels = (
                data['ct_dose_seg'].to(device),
                data['features'].to(device),
                data['label'].to(device),
            )

            # Preprocess inputs and features
            # NOTE: these operations should also be performed in lr_finder.py
            inputs = process_data.preprocess_inputs(inputs=inputs,
                                                    ct_mean=mean[0], ct_std=std[0],               ### DAN
                                                    rtdose_mean=mean[1], rtdose_std=std[1])       ### DAN)
            features = process_data.preprocess_features(features=features)

            # Make predictions
            outputs = model(x=inputs, features=features)

            

            try:
                # Cross-Entropy, Ranking, Custom
                loss = loss_function(outputs, labels)
            except:
                # BCE
                loss = loss_function(outputs, torch.reshape(labels, outputs.shape).to(outputs.dtype))

            # Evaluate model (internal validation set)
            loss_value += loss.item()
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, labels], dim=0)

            # Save patient_ids
            patient_ids += data['patient_id']

            ### DANIEL !!!
            del inputs
            del features
            del labels
            torch.cuda.empty_cache()

        # Averaging internal validation loss
        loss_value /= num_iterations
        # loss_value = round(loss_value, nr_of_decimals)

        # Compute internal validation AUC and MSE
        if loss_function_name in ['bce']:
            y_pred_list = misc.proba(sigmoid_act(y_pred))
        else:
            y_pred_list = [softmax_act(i) for i in y_pred]
        y_list = [to_onehot(i) for i in y]
        auc_value = misc.compute_auc(y_pred_list=y_pred_list, y_true_list=y_list, auc_metric=auc_metric)
        mse_value = misc.compute_mse(y_pred_list=y_pred_list, y_true_list=y_list, mse_metric=mse_metric)

        if mode.lower() == 'test':
            # Save outputs to csv
            if save_outputs:
                mode_list = ['test'] * len(patient_ids)
                misc.save_predictions(patient_ids=patient_ids, y_pred_list=y_pred_list, y_true_list=y_list,
                                      mode_list=mode_list, num_classes=num_classes, model_name=model_name,
                                      exp_dir=exp_dir, logger=logger)
            return loss_value, mse_value, auc_value, patient_ids, y_pred_list, y_list

        # Prints
        logger.my_print(f'{mode} loss: {loss_value:.3f}.')
        logger.my_print(f'{mode} AUC: {auc_value:.3f}.')
        logger.my_print(f'{mode} MSE: {mse_value:.3f}.')

        return loss_value, mse_value, auc_value


def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, optimizer_name_next, dl_class, train_dl_args_dict, logger, mean, std):
    """
    Train the model on the combined dataset, generate images, and save the images and the models.

    Args:
        model: PyTorch model
        train_dataloader: training dataloader
        val_dataloader: internal validation dataloader
        test_dataloader:
        optimizer:
        optimizer_name_next:
        dl_class:
        train_dl_args_dict:
        logger: logger

    Returns:
        None
    """
    # Initiate variables
    best_val_loss_value = np.inf
    best_val_mse_value = np.inf
    best_val_auc_value = -1
    train_loss_values_list = list() if max_epochs > 0 else [None]
    train_mse_values_list = list() if max_epochs > 0 else [None]
    train_auc_values_list = list() if max_epochs > 0 else [None]
    val_loss_values_list = list() if max_epochs > 0 else [None]
    val_mse_values_list = list() if max_epochs > 0 else [None]
    val_auc_values_list = list() if max_epochs > 0 else [None]
    test_loss_values_list = list() if max_epochs > 0 else [None]
    test_mse_values_list = list() if max_epochs > 0 else [None]
    test_auc_values_list = list() if max_epochs > 0 else [None]
    lr_list = list() if max_epochs > 0 else [None]
    batch_size_list = list() if max_epochs > 0 else [None]
    nr_epochs_not_improved = 0  # (EarlyStopping)
    nr_epochs_not_improved_lr = 0  # (LR_Finder, Reduce_LR)
    nr_epochs_not_improved_opt = 0  # (Next optimizer)

    train_num_iterations = len(train_dataloader)
    logger.my_print('Number of training iterations per epoch: {}.'.format(train_num_iterations))
    epoch = 0
    best_epoch = 0
    for epoch in range(max_epochs):
        # Print training stats
        logger.my_print(f'Epoch {epoch + 1}/{max_epochs}...')
        for param_group in optimizer.param_groups:
            logger.my_print('Learning rate: {}.'.format(param_group['lr']))
            lr_list.append(param_group['lr'])
        cur_batch_size = train_dataloader.batch_size
        logger.my_print('Batch size: {}.'.format(cur_batch_size))
        batch_size_list.append(cur_batch_size)

        # Initiate training
        model.train()
        train_loss_value = 0

        train_y_pred = torch.as_tensor([], dtype=torch.float32, device=device)
        train_y = torch.as_tensor([], dtype=torch.int8, device=device)

        for i, batch_data in tqdm(enumerate(train_dataloader)):
            # Data augmentation on all input modalities (CT, RTDOSE, segmentation)
            # NOTE: these operations should also be performed in lr_finder.py
            # AugMix currently not supported: need to fix the usage of "keys".
            # if perform_augmix:
            #     for b in range(len(train_inputs)):
            #         # Generate a random 32-bit integer
            #         seed_b = random.getrandbits(32)
            #         train_inputs[b] = process_data.aug_mix(arr=train_inputs[b],
            #                                                mixture_width=mixture_width,
            #                                                mixture_depth=mixture_depth,
            #                                                augmix_strength=augmix_strength,
            #                                                seed=seed_b)
            # else:
            # Decollate the batch data into list of dictionaries, every dictionary maps to an input data
            # seed_b = random.getrandbits(32)
            # batch_data = [process_data.data_aug(arr=i, p=data_aug_p, strength=data_aug_strength,
            #                                     seed=seed_b) for i in decollate_batch(batch_data)]
            # # Convert list to batch again
            # batch_data = list_data_collate(batch_data)

            # train_ct, train_rtdose, train_segmentation_map, train_features, train_labels = (
            #     batch_data['ct'].to(device),
            #     batch_data['rtdose'].to(device),
            #     batch_data['segmentation_map'].to(device),
            #     batch_data['features'].to(device),
            #     batch_data['label'].to(device))
            # train_inputs = torch.concat([train_ct, train_rtdose, train_segmentation_map], axis=1)
            train_inputs, train_features, train_labels = (
                batch_data['ct_dose_seg'].to(device),
                batch_data['features'].to(device),
                batch_data['label'].to(device)
            )

            # Preprocess inputs and features
            # NOTE: these operations should also be performed in lr_finder.py
            train_inputs = process_data.preprocess_inputs(inputs=train_inputs,
                                                    ct_mean=mean[0], ct_std=std[0],               ### DAN
                                                    rtdose_mean=mean[1], rtdose_std=std[1])       ### DAN
            train_features = process_data.preprocess_features(features=train_features)

            if perform_augmix:
                for b in range(len(train_inputs)):
                    # Generate a random 32-bit integer
                    seed_b = random.getrandbits(32)
                    train_inputs[b] = process_data.aug_mix(arr=train_inputs[b],
                                                           mixture_width=mixture_width,
                                                           mixture_depth=mixture_depth,
                                                           augmix_strength=augmix_strength,
                                                           device=device,
                                                           seed=seed_b)

            # Zero the parameter gradients and make predictions
            optimizer.zero_grad(set_to_none=True)
            train_outputs = model(x=train_inputs, features=train_features)

            # Calculate loss
            try:
                # Cross-Entropy, Ranking, Custom
                train_loss = loss_function(train_outputs, train_labels)
            except:
                # BCE
                train_loss = loss_function(train_outputs,
                                           torch.reshape(train_labels, train_outputs.shape).to(train_outputs.dtype))

            if config.actually_train == True:
                if optimizer_name in ['ada_hessian']:
                    # https://github.com/pytorch/pytorch/issues/4661
                    # https://discuss.pytorch.org/t/how-to-backward-the-derivative/17662?u=bpetruzzo
                    # Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient
                    # which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this.
                    # However, if we do use backward(create_graph=True), then we have to make sure to reset the
                    # .grad fields of our parameters to None after use to break the cycle and avoid the leak.
                    train_loss.backward(create_graph=True)
                    # torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
                else:
                    train_loss.backward()

            # Perform gradient clipping
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            if grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_max_norm)

            # Update model weights
            optimizer.step()

            # Reset the .grad fields of our parameters to None after use to break the cycle and avoid the memory leak
            if optimizer_name in ['ada_hessian']:
                for p in model.parameters():
                    p.grad = None

            # Scheduler: step() called after every batch update
            # Note: some schedulers generally update the LR every epoch instead of every batch, but epoch-wise update
            # will mess up LearningRateWarmUp. Batch-wise LR update is always valid and works with LearningRateWarmUp.
            if scheduler_name in ['cosine', 'exponential', 'step']:
                scheduler.step(epoch + (i + 1) / train_num_iterations)
            elif scheduler_name in ['cyclic']:
                scheduler.step()

            # Evaluate model (training set)
            train_loss_value += train_loss.item()
            train_y_pred = torch.cat([train_y_pred, train_outputs], dim=0)
            train_y = torch.cat([train_y, train_labels], dim=0)


            ### DANIEL !!!
            if (epoch + 1 == 1) or ((epoch + 1) % config.plot_interval == 0):
                pass
            else:
                del train_inputs
                del train_features
                del train_labels
            torch.cuda.empty_cache()

        # Averaging training loss
        train_loss_value /= train_num_iterations
        # train_loss_value = round(train_loss_value, nr_of_decimals)

        # Compute training AUC and MSE
        if loss_function_name in ['bce']:
            train_y_pred_list = misc.proba(sigmoid_act(train_y_pred))
        else:
            train_y_pred_list = [softmax_act(i) for i in train_y_pred]
        train_y_list = [to_onehot(i) for i in train_y]
        train_auc_value = misc.compute_auc(y_pred_list=train_y_pred_list, y_true_list=train_y_list, auc_metric=auc_metric)
        train_mse_value = misc.compute_mse(y_pred_list=train_y_pred_list, y_true_list=train_y_list, mse_metric=mse_metric)

        # Prints
        logger.my_print(f'Training loss: {train_loss_value:.3f}.')
        logger.my_print(f'Training AUC: {train_auc_value:.3f}.')
        logger.my_print(f'Training MSE: {train_mse_value:.3f}.')

        train_loss_values_list.append(train_loss_value)
        train_mse_values_list.append(train_mse_value)
        train_auc_values_list.append(train_auc_value)

        if (epoch + 1) % config.eval_interval == 0:
            # Perform internal validation
            val_loss_value, val_mse_value, val_auc_value = validate(model=model, dataloader=val_dataloader,
                                                                    mode='internal validation', logger=logger,
                                                                    mean=mean, std=std,
                                                                    save_outputs=True)
            val_loss_values_list.append(val_loss_value)
            val_mse_values_list.append(val_mse_value)
            val_auc_values_list.append(val_auc_value)

            # Perform independent testing
            test_loss_value, test_mse_value, test_auc_value = validate(model=model, dataloader=test_dataloader,
                                                                       mode='independent testing', logger=logger,
                                                                       mean=mean, std=std,
                                                                       save_outputs=True)
            test_loss_values_list.append(test_loss_value)
            test_mse_values_list.append(test_mse_value)
            test_auc_values_list.append(test_auc_value)

            # Select model
            # if val_loss_value < best_val_loss_value:
            # if val_mse_value < best_val_mse_value:
            if val_auc_value > best_val_auc_value:
                best_val_loss_value = val_loss_value
                best_val_auc_value = val_auc_value
                best_val_mse_value = val_mse_value
                best_epoch = epoch + 1
                nr_epochs_not_improved = 0
                nr_epochs_not_improved_lr = 0
                nr_epochs_not_improved_opt = 0
                torch.save(model.state_dict(), os.path.join(config.exp_dir, config.filename_best_model_pth))
                logger.my_print('Saved new best metric model.')
            else:
                nr_epochs_not_improved += 1
                nr_epochs_not_improved_lr += 1
                nr_epochs_not_improved_opt += 1

            logger.my_print(f'Best internal validation AUC: {best_val_auc_value} at epoch: {best_epoch}.')
            logger.my_print(f'Corresponding internal validation loss: {best_val_loss_value} at epoch: {best_epoch}.')
            logger.my_print(f'Corresponding internal validation MSE: {best_val_mse_value} at epoch: {best_epoch}.')

            # W&B logging
            # wandb.log({'epoch': epoch + 1,
            #            'train_loss_value': train_loss_value,
            #            'train_mse_value:': train_mse_value,
            #            'train_auc_value': train_auc_value,
            #            'val_loss_value': val_loss_value,
            #            'val_mse_value': val_mse_value,
            #            'val_auc_value': val_auc_value,
            #            'batch_size': batch_size_list[-1],
            #            'learning_rate': lr_list[-1]})

        if (epoch + 1 == 1) or ((epoch + 1) % config.plot_interval == 0):
            # Plot for random patients
            plot_inputs = train_inputs
            for patient_idx in range(min(config.max_nr_images_per_interval, len(plot_inputs))):
                arr_list = [plot_inputs[patient_idx][0].cpu(), plot_inputs[patient_idx][1].cpu(),
                            plot_inputs[patient_idx][2].cpu()]
                cmap_list = [data_preproc_config.ct_cmap, data_preproc_config.rtdose_cmap, data_preproc_config.segmentation_cmap]
                vmin_list = [0, 0, 0]
                vmax_list = [1, config.data_preproc_config.rtdose_vmax/config.rtdose_a_max, 1]
                ticks_steps_list = [1, config.data_preproc_config.rtdose_vmax/config.rtdose_a_max, 1]
                segmentation_map_list = [arr_list[2] >= 0.5, None, None]
                colorbar_title_list = [data_preproc_config.ct_colorbar_title, data_preproc_config.rtdose_colorbar_title,
                                       data_preproc_config.segmentation_colorbar_title]
                # Overlay
                overlay_list = [None, None, None]
                alpha_overlay_list = [None, None, None]
                cmap_overlay_list = [None, None, None]
                vmin_overlay_list = [None, None, None]
                vmax_overlay_list = [None, None, None]
                plot_multiple_arrs(arr_list=arr_list, overlay_list=overlay_list, alpha_overlay_list=alpha_overlay_list,
                                   nr_images=data_preproc_config.plot_nr_images_multiple,
                                   figsize=data_preproc_config.figsize_multiple,
                                   cmap_list=cmap_list, cmap_overlay_list=cmap_overlay_list,
                                   colorbar_title_list=colorbar_title_list,
                                   filename=os.path.join(config.exp_figures_dir,
                                                         'epoch_{}_idx_{}.png'.format(epoch + 1, patient_idx)),
                                   vmin_list=vmin_list, vmin_overlay_list=vmin_overlay_list,
                                   vmax_list=vmax_list, vmax_overlay_list=vmax_overlay_list,
                                   ticks_steps_list=ticks_steps_list,
                                   segmentation_map_list=segmentation_map_list)

        # EarlyStopping
        if nr_epochs_not_improved >= patience:
            logger.my_print('No internal validation improvement during the last {} consecutive epochs. '
                            'Stop training.'.format(nr_epochs_not_improved))
            return (train_loss_values_list, val_loss_values_list, test_loss_values_list, train_mse_values_list, val_mse_values_list, test_mse_values_list,
                    train_auc_values_list, val_auc_values_list, test_auc_values_list, lr_list, batch_size_list, epoch+1, best_epoch)

        # Scheduler
        if nr_epochs_not_improved_lr >= patience_lr:
            # LR_Finder scheduler
            if scheduler_name == 'lr_finder':
                # Get current learning rate (lr): use this lr if learning rate finder returns new_lr = None
                for param_group in optimizer.param_groups:
                    last_lr = param_group['lr']
                new_lr, optimizer = misc.learning_rate_finder(model=model, train_dataloader=train_dl, val_dataloader=val_dl,
                                                              optimizer=optimizer, loss_function=loss_function, data_aug_p=data_aug_p,
                                                              data_aug_strength=data_aug_strength, perform_augmix=perform_augmix,
                                                              mixture_width=mixture_width, mixture_depth=mixture_depth,
                                                              augmix_strength=augmix_strength, grad_max_norm=grad_max_norm,
                                                              device=device, logger=logger)
                if new_lr is None:
                    # Use adjusted version of old but valid lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(last_lr * config.factor_lr, config.min_lr)

            # ReduceLROnPlateau scheduler
            elif scheduler_name == 'reduce_lr':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * config.factor_lr, config.min_lr)

            # Batch_size scheduler
            elif scheduler_name == 'batch_size':
                new_batch_size = min(cur_batch_size * 2, max_batch_size)
                train_dl_args_dict['batch_size'] = new_batch_size
                train_dataloader = dl_class(**train_dl_args_dict)

            nr_epochs_not_improved_lr = 0

        # Manual_LR scheduler: LR per epoch
        if (scheduler_name == 'manual_lr') and (len(manual_lr) > 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = manual_lr[0]
            manual_lr.pop(0)

        # Next optimizer
        if nr_epochs_not_improved_opt >= patience_opt:
            # Consider new optimizer
            if len(optimizer_name_next) >= 1:
                optimizer_name_i = optimizer_name_next[0]
                optimizer_name_next.pop(0)
                # Get current learning rate
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                optimizer = misc.get_optimizer(optimizer_name=optimizer_name_i, model=model, lr=lr, momentum=momentum,
                                               weight_decay=weight_decay, hessian_power=hessian_power,
                                               num_batches_per_epoch=num_batches_per_epoch,
                                               use_lookahead=use_lookahead, lookahead_k=lookahead_k,
                                               lookahead_alpha=lookahead_alpha, logger=logger)
                scheduler.optimizer = optimizer
                logger.my_print('Next optimizer: {}.'.format(optimizer_name_i))

            nr_epochs_not_improved_opt = 0

    return (train_loss_values_list, val_loss_values_list, test_loss_values_list, train_mse_values_list, val_mse_values_list, test_mse_values_list,
            train_auc_values_list, val_auc_values_list, test_auc_values_list, lr_list, batch_size_list, epoch+1, best_epoch)


if __name__ == '__main__':
    # Make sure that data_preproc_config.use_umcg == True
    use_umcg = data_preproc_config.use_umcg
    assert use_umcg

    # Initialize variables
    wandb_mode = config.wandb_mode
    torch_version = config.torch_version
    seed = config.seed
    device = config.device
    cudnn_benchmark = config.cudnn_benchmark
    exp_dir = config.exp_dir
    filename_stratified_sampling_test_csv = config.filename_stratified_sampling_test_csv
    filename_stratified_sampling_full_csv = config.filename_stratified_sampling_full_csv
    perform_stratified_sampling_full = config.perform_stratified_sampling_full
    sampling_type = config.sampling_type
    cv_folds = config.cv_folds
    cv_type = config.cv_type
    use_sampler = config.use_sampler
    perform_data_aug = config.perform_data_aug
    rand_cropping_size = config.rand_cropping_size
    input_size = config.input_size
    resize_mode = config.resize_mode
    seg_orig_labels = config.seg_orig_labels
    seg_target_labels = config.seg_target_labels
    # Deep Learning model
    dataloader_drop_last = config.dataloader_drop_last
    data_aug_p = config.data_aug_p
    data_aug_strength = config.data_aug_strength
    perform_augmix = config.perform_augmix
    mixture_depth = config.mixture_depth
    mixture_width = config.mixture_width
    augmix_strength = config.augmix_strength
    model_name = config.model_name
    features_dl = config.features_dl
    resnet_shortcut = config.resnet_shortcut
    filters = config.filters
    kernel_sizes = config.kernel_sizes
    strides = config.strides
    pad_value = config.pad_value
    lrelu_alpha = config.lrelu_alpha
    pooling_conv_filters = config.pooling_conv_filters
    perform_pooling = config.perform_pooling
    linear_units = config.linear_units
    dropout_p = config.dropout_p
    use_bias = config.use_bias
    num_classes = config.num_classes
    num_ohe_classes = config.num_ohe_classes
    pretrained_path = config.pretrained_path
    weight_init_name = config.weight_init_name
    kaiming_a = config.kaiming_a
    kaiming_mode = config.kaiming_mode
    kaiming_nonlinearity = config.kaiming_nonlinearity
    gain = config.gain
    optimizer_name = config.optimizer_name
    optimizer_name_next = config.optimizer_name_next
    loss_function_name = config.loss_function_name
    manual_lr = config.manual_lr
    base_lr = config.lr
    momentum = config.momentum
    weight_decay = config.weight_decay
    hessian_power = config.hessian_power
    use_lookahead = config.use_lookahead
    lookahead_k = config.lookahead_k
    lookahead_alpha = config.lookahead_alpha
    lr_finder_num_iter = config.lr_finder_num_iter
    scheduler_name = config.scheduler_name
    warmup_batches = config.warmup_batches
    T_0 = config.T_0
    T_mult = config.T_mult
    eta_min = config.eta_min
    step_size_up = config.step_size_up
    gamma = config.gamma
    step_size = config.step_size
    grad_max_norm = config.grad_max_norm
    label_weights = config.label_weights
    label_smoothing = config.label_smoothing
    loss_weights = config.loss_weights
    nr_of_decimals = config.nr_of_decimals
    # max_epochs = config.max_epochs
    batch_size = config.batch_size
    max_batch_size = config.max_batch_size
    patience_lr = config.patience_lr
    patience_opt = config.patience_opt
    patience = config.patience
    # Logistic regression model
    patient_id_col = data_preproc_config.patient_id_col
    features_lr = data_preproc_config.features
    endpoint = data_preproc_config.endpoint
    patient_id_length = data_preproc_config.patient_id_length
    ijk = 0

    sigmoid_act = torch.nn.Sigmoid()
    softmax_act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=num_ohe_classes)
    mse_metric = MSEMetric()
    auc_metric = ROCAUCMetric()

    lmn = 0
    excl = ''
    
    with torch.no_grad():
      torch.cuda.empty_cache()
    
    for seed in [4]:
        for perform_stratified_sampling_full, filename_stratified_sampling_test_csv, filename_stratified_sampling_full_csv in zip(
                [False],
                [filename_stratified_sampling_test_csv],
                [filename_stratified_sampling_full_csv]
        ):

            for features_dl in [
                # ['HN35_Xerostomia_W01_not_at_all', 'HN35_Xerostomia_W01_little', 'HN35_Xerostomia_W01_moderate_to_severe'],
                ['HN35_Xerostomia_W01_not_at_all', 'HN35_Xerostomia_W01_little',
                 'HN35_Xerostomia_W01_moderate_to_severe', 'Sex', 'Age'],
            ]:
                for ijk, seg_target_labels in enumerate([
                    # [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],  # 31-10-2022 (strata_manual_ds_old_bsl): VERY GOOD WITH BSL_3, 1, 3, 3, [1, 3]: AUC = 0.8, 0.79)
                    #
                    # [0, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5],
                    #
                    # [0, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1],  # good

                    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0.1, 0, 0.1, 0.1, 0, 0, 0, 0],  # 9-11-2022 (strata_manual_ds_old_seg_labels): good with (BSL_3 | BSL_5) with augmix_strength = 3 (AUC = 0.79, 0.79).
                    # 7-9-2022 (strata_manual_ds_old_augmix): BSL_5 with augmix_strength = 3 (AUC = 0.79, 0.79).
                ]):
                    perform_data_aug = True
                    perform_augmix = True
                    data_aug_strength = 1
                    augmix_strength = 3

                    # for optimizer_name in ['ada_belief', 'ada_bound', 'ada_mod', 'adam', 'madgrad']:

                    # RUN 2: repeat this for '_excl_12' and '_excl_2'
                    excl = ''
                    # for model_name, optimizer_name in zip(
                    #     ['dcnn_dws_lrelu', 'dcnn_dws_lrelu', 'dcnn_dws_lrelu', 'dcnn_lrelu'],
                    #     ['ada_belief', 'ada_bound', 'adam', 'ada_bound']
                    # ):
                    model_name = 'dcnn_lrelu'
                    # TODO: TEMPORARY: PRETRAINING
                    pretrained_models_path = 'pretrained_models/dcnn/'
                    label_smoothing = 0
                    optimizer_name = 'ada_bound'
                    scheduler_name = 'cosine'
                    label_weights = [1, 3]
                    T_0 = 8
                    gamma = 0.95
                    use_sampler = False
                    loss_function_name = 'custom' 
                    loss_weights = [1, 0, 0, 0, 0, 0]  # list of weight for [ce, dice, f1, l1, ranking, soft_auc].
                    
                    for label_weights, loss_weights in zip(
                        [                            
                            
                            [1, 1.5],
                            #[1, 2],
                        ],
                        [                            
                            
                            [1, 0, 1, 1, 0, 0],
                            #[1, 0, 1, 1, 0, 0],
                            
                    ]):
                        # excl = '_excl_12'
                        # for model_name, optimizer_name in zip(
                        #     ['dcnn_dws_lrelu', 'dcnn_dws_lrelu', 'dcnn_dws_lrelu'],
                        #     ['ada_bound', 'adam', 'ada_belief']
                        # ):

                        # excl = '_excl_2'
                        # for model_name, optimizer_name in zip(
                        #     ['dcnn_dws_lrelu', 'dcnn_dws_lrelu', 'dcnn_lrelu'],
                        #     ['ada_bound', 'ada_belief', 'ada_bound']
                        # ):

                        # for excl in ['_excl_2']:  # ['', '_excl_12', '_excl_2']:
                        if excl == '_excl_12':
                            data_preproc_config.filename_exclude_patients_csv = 'exclude_patients_metal_artifact_12.csv'
                        elif excl == '_excl_2':
                            data_preproc_config.filename_exclude_patients_csv = 'exclude_patients_metal_artifact_2.csv'
                        else:
                            data_preproc_config.filename_exclude_patients_csv = 'exclude_patients_umcg.csv'

                        for cv_folds, seed in zip([1], [17]):

                            # for optimizer_name in ['adam', 'rmsprop', 'acc_sgd', 'ada_belief', 'ada_bound',
                            #                        'ada_bound_w', 'ada_mod', 'apollo', 'diff_grad', 'madgrad',
                            #                        'novo_grad', 'pid', 'qh_adam', 'qhm', 'r_adam',
                            #                        'ranger_qh', 'ranger_21', 'swats', 'yogi']:

                            lmn += 1
                            if lmn >= 0:
                                # Initialize W&B
                                # run = wandb.init(project='Xerostomia', reinit=True, mode=wandb_mode)

                                # Set torch.backends.cudnn.benchmark after set_determinism(), because set_determinism()
                                # sets torch.backends.cudnn.benchmark to False
                                torch.manual_seed(seed=seed)
                                set_determinism(seed=seed)
                                random.seed(a=seed)
                                np.random.seed(seed=seed)
                                torch.backends.cudnn.benchmark = cudnn_benchmark

                                # Initialize variables
                                train_auc_list, train_auc_lr_list = list(), list()
                                val_auc_list, val_auc_lr_list = list(), list()
                                train_auc_value, val_auc_value, test_auc_value = [None] * 3
                                train_auc_list_mean, val_auc_list_mean = [None] * 2
                                test_y_pred_sum, test_y_sum = [None] * 2
                                test_y_pred_lr_sum, test_y_lr_sum = [None] * 2
                                if model_name in ['efficientnet-b{}'.format(i) for i in range(9)] + \
                                        ['efficientnetv2_{}'.format(i) for i in ['s', 'm']] + \
                                        ['resnet_original' + 'resnext_original']:
                                    # Only batch_size > 1 is allowed for batch_norm
                                    assert batch_size > 1
                                    dataloader_drop_last = True

                                for fold in range(cv_folds):
                                    # Initialize variables
                                    # best_val_auc_idx = 0
                                    logger = initialize(torch_version=torch_version, device=device)
                                    logger.my_print('Model name: {}'.format(model_name))
                                    logger.my_print('Fold: {}'.format(fold))
                                    logger.my_print('Seed: {}'.format(seed))
                                    test_loss_value, test_mse_value, test_auc_value, test_auc_mean, test_auc_lr, test_auc_lr_mean = [None] * 6
                                    # TODO: TEMPORARY: PRETRAINING
                                    pretrained_path = os.path.join(pretrained_models_path, str(fold),
                                                                   config.filename_best_model_pth)
                                    start = time.time()

                                    # Cross-Validation
                                    if cv_folds > 1:
                                        if fold == 0:
                                            # Fetch training, internal validation and test files
                                            cv_dict_0, cv_dict_1, test_dict = load_data.get_files(
                                                sampling_type=sampling_type, features=features_dl,
                                                filename_stratified_sampling_test_csv=filename_stratified_sampling_test_csv,
                                                filename_stratified_sampling_full_csv=filename_stratified_sampling_full_csv,
                                                perform_stratified_sampling_full=perform_stratified_sampling_full,
                                                seed=seed, logger=logger)

                                            # Combine training and internal validation files
                                            cv_dict = cv_dict_0 + cv_dict_1

                                            # Perform Stratified CV
                                            if cv_type == 'stratified':
                                                logger.my_print('Performing stratified {}-fold CV.'.format(cv_folds))
                                                # Create stratified CV class
                                                cv_object = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                                                # Determine endpoint value to stratify on
                                                cv_y = [x['label'] for x in cv_dict]
                                                # For each fold: determine train and val indices
                                                cv_idx_list = list()
                                                # Source: https://www.programcreek.com/python/example/91147/sklearn.model_selection.StratifiedKFold
                                                # Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
                                                # Note that providing y is sufficient to generate the splits and hence
                                                # np.zeros(len(cv_y)) may be used as a placeholder for X instead of actual
                                                # training data.
                                                for idx in cv_object.split(X=cv_dict, y=cv_y):
                                                    cv_idx_list.append(idx)

                                        # For this fold: select training and internal validation indices
                                        train_idx, valid_idx = cv_idx_list[fold]

                                        # Fetch training set using cv indices in 'cv_idx_list'
                                        train_dict = list()
                                        for i in train_idx:
                                            train_dict.append(cv_dict[i])

                                        # Fetch internal validation set using cv indices in 'cv_idx_list'
                                        val_dict = list()
                                        for i in valid_idx:
                                            val_dict.append(cv_dict[i])
                                        assert len(train_dict) + len(val_dict) == len(cv_dict)

                                    # Training-validation-test split
                                    else:
                                        train_dict, val_dict, test_dict = [None] * 3

                                    (train_dl, val_dl, test_dl, train_ds, dl_class, train_dl_args_dict, batch_size, channels,
                                    depth, height, width, n_train_0, n_train_1, n_val_0, n_val_1, n_test_0, n_test_1,
                                    n_features, train_dict, val_dict, test_dict, norm_mean, norm_std) = (
                                        load_data.main(train_dict=train_dict, val_dict=val_dict, test_dict=test_dict,
                                                       features=features_dl, perform_data_aug=perform_data_aug, data_aug_p=data_aug_p,
                                                       data_aug_strength=data_aug_strength, rand_cropping_size=rand_cropping_size,
                                                       input_size=input_size, resize_mode=resize_mode, seg_orig_labels=seg_orig_labels,
                                                       seg_target_labels=seg_target_labels, batch_size=batch_size,
                                                       use_sampler=use_sampler, fold=fold, drop_last=dataloader_drop_last,
                                                       sampling_type=sampling_type,
                                                       filename_stratified_sampling_test_csv=filename_stratified_sampling_test_csv,
                                                       filename_stratified_sampling_full_csv=filename_stratified_sampling_full_csv,
                                                       perform_stratified_sampling_full=perform_stratified_sampling_full,
                                                       seed=seed, device=device, logger=logger))
                                    logger.my_print('Input shape: {}.'.format([batch_size, channels, depth, height, width]))
                                    num_batches_per_epoch = len(train_dl)

                                    # Initialize model
                                    print(pretrained_path)
                                    model = misc.get_model(model_name=model_name, num_ohe_classes=num_ohe_classes, channels=channels,
                                        depth=depth, height=height, width=width, n_features=n_features,
                                        resnet_shortcut=resnet_shortcut,
                                        num_classes=num_classes, filters=filters, kernel_sizes=kernel_sizes, strides=strides,
                                        pad_value=pad_value,
                                        lrelu_alpha=lrelu_alpha, dropout_p=dropout_p, pooling_conv_filters=pooling_conv_filters,
                                        perform_pooling=perform_pooling, linear_units=linear_units,
                                        clinical_variables_position=config.clinical_variables_position,   ### DANIEL
                                        clinical_variables_linear_units=config.clinical_variables_linear_units,   ### DANIEL
                                        clinical_variables_dropout_p=config.clinical_variables_dropout_p,   ### DANIEL
                                        use_bias=use_bias, pretrained_path= pretrained_path,      ### DANIEL
                                        logger=logger)
        
                                    model.to(device=device)

                                    # Compile model (PyTorch 2)
                                    """
                                    if torch_version.startswith('2.'):
                                        model = torch.compile(model)
                                    """

                                    # Weight initialization
                                    if 'selu' in model_name:
                                        # Source: https://github.com/bioinf-jku/SNNs/tree/master/Pytorch
                                        weight_init_name = 'kaiming_normal'
                                        kaiming_a = None
                                        kaiming_mode = 'fan_in'
                                        kaiming_nonlinearity = 'linear'

                                    logger.my_print('Weight init name: {}.'.format(weight_init_name))
                                    if pretrained_path is not None:
                                        logger.my_print('Using pretrained weights: {}.'.format(pretrained_path))
                                    elif ((pretrained_path is None) and
                                            (weight_init_name is not None) and
                                            ('efficientnet' not in model_name) and
                                            ('convnext' not in model_name) and
                                            ('resnext' not in model_name)):
                                        # Source: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
                                        model.apply(lambda m: weights_init(m=m,
                                                                           weight_init_name=weight_init_name,
                                                                           kaiming_a=kaiming_a,
                                                                           kaiming_mode=kaiming_mode,
                                                                           kaiming_nonlinearity=kaiming_nonlinearity,
                                                                           gain=gain,
                                                                           logger=logger))
                                        logger.my_print('Using our own weights init scheme for {}.'.format(model_name))
                                    else:
                                        logger.my_print('Using default PyTorch weights init scheme for {}.'.format(model_name))

                                    # Create and save model summary
                                    # summary() only accepts non-zero integers, i.e. n_features = 0 is invalid in `input_size`.
                                    # However n_features = 0 in get_model() will make `n_features` in summary() redundant, so
                                    # we just use `1` in the input of summary().
                                    # Important note: torchinfo.summary() somehow uses some random() on input_size, because it will create
                                    # random input. I.e. input_size changes the next random values.
                                    # total_params = misc.get_model_summary(model=model,
                                    #                                       input_size=[(batch_size, channels, depth, height, width),
                                    #                                                   (batch_size, max(n_features, 1))],
                                    #                                       device=device,
                                    #                                       logger=logger)
                                    total_params = 0

                                    # Initialize loss function
                                    loss_function = misc.get_loss_function(
                                        loss_function_name=loss_function_name,
                                        label_weights=label_weights,
                                        label_smoothing=label_smoothing,
                                        loss_weights=loss_weights, device=device)

                                    # Collect model's gradients and topology
                                    # wandb.watch(model, loss_function, log_freq=1, log='all')

                                    # Initialize optimizer
                                    if scheduler_name == 'manual_lr':
                                        lr = manual_lr[0]
                                        manual_lr.pop(0)
                                    else:
                                        lr = base_lr
                                    optimizer = misc.get_optimizer(optimizer_name=optimizer_name, model=model, lr=lr, momentum=momentum,
                                                                   weight_decay=weight_decay, hessian_power=hessian_power,
                                                                   num_batches_per_epoch=num_batches_per_epoch,
                                                                   use_lookahead=use_lookahead, lookahead_k=lookahead_k,
                                                                   lookahead_alpha=lookahead_alpha, logger=logger)

                                    # Perform learning rate finder
                                    if lr_finder_num_iter > 0 and (config.max_epochs > 0):
                                        lr, optimizer = misc.learning_rate_finder(
                                            model=model, train_dataloader=train_dl, val_dataloader=val_dl, optimizer=optimizer,
                                            optimizer_name=optimizer_name, loss_function=loss_function, data_aug_p=data_aug_p,
                                            data_aug_strength=data_aug_strength, perform_augmix=perform_augmix,
                                            mixture_width=mixture_width, mixture_depth=mixture_depth,
                                            augmix_strength=augmix_strength, grad_max_norm=grad_max_norm, lr_finder_num_iter=lr_finder_num_iter,
                                            save_filename=config.filename_lr_finder_png, device=device, logger=logger)
                                    starting_lr = lr

                                    # If lr = None (i.e. lr cannot be found), then skip iteration of for-loop
                                    if lr is None:
                                        # continue
                                        max_epochs = 0
                                    else:
                                        max_epochs = config.max_epochs

                                    # Initialize scheduler
                                    # Note: should be performed after learning_rate_finder()
                                    scheduler = misc.get_scheduler(scheduler_name=scheduler_name, optimizer=optimizer,
                                                                   optimal_lr=lr, T_0=T_0, T_mult=T_mult,
                                                                   eta_min=eta_min, step_size_up=step_size_up,
                                                                   gamma=gamma, step_size=step_size,
                                                                   warmup_batches=warmup_batches,
                                                                   num_batches_per_epoch=num_batches_per_epoch,
                                                                   lr_finder_num_iter=lr_finder_num_iter, logger=logger)

                                    # Train model
                                    (train_loss_values, val_loss_values, test_loss_values, train_mse_values, val_mse_values, test_mse_values,
                                     train_auc_values, val_auc_values, test_auc_values, lr_values, batch_size_values, epochs, best_epoch) = \
                                        train(model=model, train_dataloader=train_dl, val_dataloader=val_dl, test_dataloader=test_dl,
                                              optimizer=optimizer, optimizer_name_next=optimizer_name_next,
                                              dl_class=dl_class, train_dl_args_dict=train_dl_args_dict,
                                              logger=logger, mean=norm_mean, std=norm_std)

                                    # best_val_auc_idx = best_epoch - 1  # alternative: val_auc_values.index(max(val_auc_values))
                                    # assert max(val_auc_values) == val_auc_values[best_val_auc_idx]

                                    if max_epochs >= 0:
                                        # Plot training and internal validation losses
                                        y_list = [[train_loss_values, val_loss_values],
                                                  [train_mse_values, val_mse_values],
                                                  [train_auc_values, val_auc_values], [lr_values],
                                                  [batch_size_values]]
                                        y_label_list = ['Loss', 'MSE', 'AUC', 'LR', 'Batch_size']
                                        legend_list = [['Training', 'Internal validation']] * (
                                                len(y_list) - 2) + [None, None]
                                        misc.plot_values(y_list=y_list, y_label_list=y_label_list,
                                                         best_epoch=best_epoch, legend_list=legend_list,
                                                         figsize=config.figsize, save_filename=os.path.join(exp_dir,
                                                                                                            config.filename_results_png))

                                        # Load best model
                                        model.load_state_dict(torch.load(os.path.join(exp_dir, config.filename_best_model_pth)))

                                        # Store predictions on training and validation set
                                        # IMPORTANT NOTE: train_dl still performs data augmentation!
                                        train_loss_value, train_mse_value, train_auc_value, train_patient_ids, train_y_pred_list, train_y_list = validate(
                                            model=model, dataloader=train_dl, mode='test', logger=logger, 
                                                                        mean=norm_mean, std=norm_std,
                                                                       save_outputs=True)
                                        val_loss_value, val_mse_value, val_auc_value, val_patient_ids, val_y_pred_list, val_y_list = validate(
                                            model=model, dataloader=val_dl, mode='test', logger=logger, 
                                                                        mean=norm_mean, std=norm_std,
                                                                       save_outputs=True)
                                        all_patient_ids = train_patient_ids + val_patient_ids
                                        all_y_pred_list = train_y_pred_list + val_y_pred_list
                                        all_y_list = train_y_list + val_y_list
                                        all_modes = ['train'] * len(train_patient_ids) + ['val'] * len(val_patient_ids)

                                        train_auc_list.append(train_auc_value)
                                        val_auc_list.append(val_auc_value)

                                        # Test evaluate model
                                        if test_dl is not None:
                                            test_loss_value, test_mse_value, test_auc_value, test_patient_ids, test_y_pred_list, test_y_list = validate(
                                                model=model, dataloader=test_dl, mode='test', logger=logger,
                                                mean=norm_mean, std=norm_std,
                                                save_outputs=True)
                                            

                                            # Store predictions on test set
                                            all_patient_ids += test_patient_ids
                                            all_y_pred_list += test_y_pred_list
                                            all_y_list += test_y_list
                                            all_modes += ['test'] * len(test_patient_ids)

                                            if (test_y_pred_sum is None) and (test_y_sum is None):
                                                test_y_pred_sum = test_y_pred_list
                                                test_y_sum = test_y_list
                                            else:
                                                test_y_pred_sum = [x + y for x, y in zip(test_y_pred_sum, test_y_pred_list)]
                                                test_y_sum = [x + y for x, y in zip(test_y_sum, test_y_list)]

                                        # Save predictions on training, validation and test set
                                        misc.save_predictions(patient_ids=all_patient_ids, y_pred_list=all_y_pred_list,
                                                              y_true_list=all_y_list, mode_list=all_modes,
                                                              num_classes=num_classes, model_name=model_name + '_all',
                                                              exp_dir=exp_dir, logger=logger)

                                    # Run logistic regression (LR) model (reference model)
                                    features_csv = os.path.join(config.data_dir, data_preproc_config.filename_features_csv)
                                    patient_ids_json = os.path.join(exp_dir, config.filename_train_val_test_patient_ids_json.format(fold))
                                    (train_patient_ids_lr, train_y_pred_lr, train_y_lr,
                                     val_patient_ids_lr, val_y_pred_lr, val_y_lr,
                                     test_patient_ids_lr, test_y_pred_lr, test_y_lr, lr_coefficients) = (
                                        run_logistic_regression(features_csv=features_csv, patient_id_col=patient_id_col,
                                                                baseline_col=data_preproc_config.baseline_col,
                                                                submodels_features=data_preproc_config.submodels_features, features=features_lr,
                                                                lr_coefficients=data_preproc_config.lr_coefficients, endpoint=endpoint,
                                                                patient_id_length=patient_id_length, patient_ids_json=patient_ids_json,
                                                                seed=seed, nr_of_decimals=nr_of_decimals, logger=logger))

                                    # Compute training and internal validation AUC of LR model
                                    train_y_pred_lr_list = [x for x in train_y_pred_lr]
                                    train_y_lr_list = [to_onehot(i) for i in train_y_lr]
                                    train_auc_lr = misc.compute_auc(y_pred_list=train_y_pred_lr_list, y_true_list=train_y_lr_list,
                                                                    auc_metric=auc_metric)
                                    train_auc_lr_list.append(train_auc_lr)

                                    val_y_pred_lr_list = [x for x in val_y_pred_lr]
                                    val_y_lr_list = [to_onehot(i) for i in val_y_lr]
                                    val_auc_lr = misc.compute_auc(y_pred_list=val_y_pred_lr_list, y_true_list=val_y_lr_list, auc_metric=auc_metric)
                                    val_auc_lr_list.append(val_auc_lr)

                                    # Store predictions on training and validation set of LR model
                                    # IMPORTANT NOTE: train_dl still performs data augmentation!
                                    all_patient_ids_lr = train_patient_ids_lr + val_patient_ids_lr
                                    all_y_pred_lr_list = train_y_pred_lr_list + val_y_pred_lr_list
                                    all_y_lr_list = train_y_lr_list + val_y_lr_list
                                    all_modes_lr = ['train'] * len(train_patient_ids_lr) + ['val'] * len(val_patient_ids_lr)

                                    # Test evaluate (LR model)
                                    if test_dl is not None:
                                        test_y_pred_lr_list = [x for x in test_y_pred_lr]
                                        test_y_lr_list = [to_onehot(i) for i in test_y_lr]
                                        test_auc_lr = misc.compute_auc(y_pred_list=test_y_pred_lr_list, y_true_list=test_y_lr_list, auc_metric=auc_metric)

                                        # Determine patient_ids in test set (for misc.save_predictions())
                                        mode_lr_list = ['test'] * len(test_patient_ids_lr)

                                        all_patient_ids_lr += test_patient_ids_lr
                                        all_y_pred_lr_list += test_y_pred_lr_list
                                        all_y_lr_list += test_y_lr_list
                                        all_modes_lr += mode_lr_list

                                        # Save outputs to csv
                                        misc.save_predictions(patient_ids=test_patient_ids_lr, y_pred_list=test_y_pred_lr_list, y_true_list=test_y_lr_list,
                                                              mode_list=mode_lr_list, num_classes=num_classes, model_name='lr', exp_dir=exp_dir, logger=logger)

                                        if (test_y_pred_lr_sum is None) and (test_y_lr_sum is None):
                                            test_y_pred_lr_sum = test_y_pred_lr_list
                                            test_y_lr_sum = test_y_lr_list
                                        else:
                                            test_y_pred_lr = [x for x in test_y_pred_lr_list]
                                            test_y_pred_lr_sum = [x + y for x, y in zip(test_y_pred_lr_sum, test_y_pred_lr_list)]
                                            test_y_lr_sum = [x + y for x, y in zip(test_y_lr_sum, test_y_lr_list)]

                                        # Ensemble (DL and LR model)
                                        if (fold == cv_folds - 1) and (cv_folds > 1):
                                            # Make sure that test_y_sum and test_y_lr_sum have only `num_classes` different tensors:
                                            # e.g. cv_folds = 5: {(0.0, 5.0), (5.0, 0.0)}
                                            if max_epochs >= 0:                          ### DANIEL
                                                assert len(set([tuple(x.tolist()) for x in test_y_sum])) == num_classes
                                                test_y_pred_ens = [x / cv_folds for x in test_y_pred_sum]
                                                test_y_ens = [x / cv_folds for x in test_y_sum]
                                                misc.save_predictions(patient_ids=test_patient_ids, y_pred_list=test_y_pred_ens, y_true_list=test_y_ens,
                                                                      mode_list=mode_lr_list, num_classes=num_classes, model_name=model_name + '_ens',
                                                                      exp_dir=exp_dir, logger=logger)

                                            assert len(set([tuple(x.tolist()) for x in test_y_lr_sum])) == num_classes
                                            test_y_pred_lr_ens = [x / cv_folds for x in test_y_pred_lr_sum]
                                            test_y_lr_ens = [x / cv_folds for x in test_y_lr_sum]
                                            misc.save_predictions(patient_ids=test_patient_ids_lr, y_pred_list=test_y_pred_lr_ens, y_true_list=test_y_lr_ens,
                                                                  mode_list=mode_lr_list, num_classes=num_classes, model_name='lr_ens',
                                                                  exp_dir=exp_dir, logger=logger)

                                    # Save predictions on training, validation and test set
                                    misc.save_predictions(patient_ids=all_patient_ids_lr,
                                                          y_pred_list=all_y_pred_lr_list,
                                                          y_true_list=all_y_lr_list, mode_list=all_modes_lr,
                                                          num_classes=num_classes,
                                                          model_name='lr_all',
                                                          exp_dir=exp_dir, logger=logger)

                                    if max_epochs >= 0:                          ### DANIEL
                                        # Create results.csv
                                        # best_val_auc_idx = val_auc_values.index(max(val_auc_values))  # alternative: best_epoch (- 1)
                                        # assert max(val_auc_values) == val_auc_values[best_val_auc_idx]
                                        misc.create_results(train_loss=train_loss_value, val_loss=val_loss_value, test_loss=test_loss_value,
                                                            train_auc=train_auc_value, train_auc_lr=train_auc_lr,
                                                            val_auc=val_auc_value, val_auc_lr=val_auc_lr,
                                                            test_auc=test_auc_value, test_auc_lr=test_auc_lr,
                                                            seed=seed, epochs=epochs, best_epoch=best_epoch,
                                                            batch_size=batch_size, starting_lr=starting_lr, patience=patience, model_name=model_name,
                                                            features_dl=features_dl, n_features=n_features, filters=filters, kernel_sizes=kernel_sizes,
                                                            strides=strides, lrelu_alpha=lrelu_alpha, pooling_conv_filters=pooling_conv_filters,
                                                            linear_units=linear_units, dropout_p=dropout_p, use_bias=use_bias,
                                                            total_params=total_params, loss_weights=loss_weights,
                                                            n=n_train_0 + n_train_1 + n_val_0 + n_val_1 + n_test_0 + n_test_1,
                                                            n_train=[n_train_0, n_train_1], n_val=[n_val_0, n_val_1], n_test=[n_test_0, n_test_1],
                                                            train_val_test_frac=[config.train_frac, config.val_frac, 1-config.train_frac-config.val_frac],
                                                            cv_folds=cv_folds, input_shape=[batch_size, channels, depth, height, width],
                                                            ct_b_min_max=[config.ct_b_min, config.ct_b_max],
                                                            rtdose_b_min_max=[config.rtdose_b_min, config.rtdose_b_max],
                                                            seg_orig_labels=seg_orig_labels, seg_target_labels=seg_target_labels,
                                                            # d_contra_0_max=config.d_contra_0_max,
                                                            interpolation_mode_3d=[config.ct_interpol_mode_3d,
                                                                                   config.rtdose_interpol_mode_3d,
                                                                                   config.segmentation_interpol_mode_3d],
                                                            interpolation_mode_2d=[config.ct_interpol_mode_2d,
                                                                                   config.rtdose_interpol_mode_2d,
                                                                                   config.segmentation_interpol_mode_2d],
                                                            perform_augmix=perform_augmix, weight_init_name=weight_init_name,
                                                            optimizer=optimizer_name, optimizer_name_next=optimizer_name_next,
                                                            loss_function=loss_function_name, label_weights=label_weights,
                                                            label_smoothing=label_smoothing, lr_finder_num_iter=lr_finder_num_iter,
                                                            scheduler=scheduler_name, grad_max_norm=grad_max_norm,
                                                            lr_coefficients=lr_coefficients)

                                    end = time.time()
                                    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, nr_of_decimals)))
                                    logger.my_print('DONE!')
                                    logger.close()

                                    """
                                    # Rename folder
                                    src_folder_name = os.path.join(exp_dir)
                                    print("SOURCE FOLDER NAME")
                                    print(src_folder_name)

                                    if (fold == cv_folds - 1) and (cv_folds > 1):
                                        dst_folder_name = os.path.join(exp_dir + '_{}'.format(lmn) +
                                                                       '_{}'.format(fold) +
                                                                       '_{}'.format(best_epoch) +
                                                                       '_{}'.format(label_weights) +
                                                                       '_{}'.format(loss_weights) +
                                                                       '_auc' +
                                                                       '_tr_{}'.format(train_auc_value) +
                                                                       '_lr_{}'.format(train_auc_lr) +
                                                                       '_val_{}'.format(val_auc_value) +
                                                                       '_lr_{}'.format(val_auc_lr)
                                                                       )
                                    else:
                                        dst_folder_name = os.path.join(exp_dir + '_{}'.format(seed) +
                                                                       '_bsl_{}'.format(len(features_dl)) +
                                                                       excl + '_{}'.format(lmn) +
                                                                       '_{}'.format(fold) + '_{}'.format(ijk) +
                                                                       '_{}'.format(best_epoch) +
                                                                       '_{}'.format(label_weights) +
                                                                       '_{}'.format(loss_weights) +
                                                                       '_{}'.format(data_aug_strength) +
                                                                       '_{}'.format(augmix_strength) +
                                                                       '_{}'.format(model_name) +
                                                                       '_params_{}'.format(total_params) +
                                                                       '_{}'.format(optimizer_name) +
                                                                       '_{}'.format(scheduler_name) +
                                                                       '_auc' +
                                                                       '_tr_{}'.format(train_auc_value) +
                                                                       '_lr_{}'.format(train_auc_lr) +
                                                                       '_val_{}'.format(val_auc_value) +
                                                                       '_lr_{}'.format(val_auc_lr)
                                                                       )

                                    if test_dl is not None:
                                        dst_folder_name += '_test_{}'.format(test_auc_value)
                                        dst_folder_name += '_lr_{}'.format(test_auc_lr)

                                    # Results of all folds (avg and ensemble)
                                    if (fold == cv_folds - 1) and (cv_folds > 1):
                                        dst_folder_name += '_avg'

                                        if max_epochs > 0:
                                            # DL training (avg)
                                            train_auc_list = [x for x in train_auc_list if x is not None]
                                            train_auc_list_mean = round(sum(train_auc_list) / len(train_auc_list),
                                                                        nr_of_decimals)

                                        # LR training (avg)
                                        train_auc_lr_list = [x for x in train_auc_lr_list if x is not None]
                                        train_auc_lr_list_mean = round(sum(train_auc_lr_list) / len(train_auc_lr_list),
                                                                       nr_of_decimals)
                                        dst_folder_name += '_tr_{}_lr_{}'.format(train_auc_list_mean, train_auc_lr_list_mean)

                                        # Internal validation (avg) and test (ensemble)
                                        # DL model
                                        dst_folder_name += '_val'
                                        if max_epochs > 0:
                                            # Validation (avg)
                                            val_auc_list = [x for x in val_auc_list if x is not None]
                                            val_auc_list_mean = round(sum(val_auc_list) / len(val_auc_list), nr_of_decimals)

                                            # Test (ensemble)
                                            test_auc_mean = misc.compute_auc(y_pred_list=test_y_pred_ens,
                                                                             y_true_list=test_y_ens,
                                                                             auc_metric=auc_metric)
                                        dst_folder_name += '_{}'.format(val_auc_list_mean)

                                        # LR model
                                        # Validation (avg)
                                        val_auc_lr_list = [x for x in val_auc_lr_list if x is not None]
                                        val_auc_lr_list_mean = round(sum(val_auc_lr_list) / len(val_auc_lr_list),
                                                                     nr_of_decimals)
                                        dst_folder_name += '_lr_{}'.format(val_auc_lr_list_mean)
                                        # Test (ensemble)
                                        test_auc_lr_mean = misc.compute_auc(y_pred_list=test_y_pred_lr_ens,
                                                                            y_true_list=test_y_lr_ens,
                                                                            auc_metric=auc_metric)

                                        # Ensemble
                                        dst_folder_name += '_ens_{}'.format(test_auc_mean)
                                        dst_folder_name += '_lr_{}'.format(test_auc_lr_mean)

                                    shutil.move(src_folder_name, dst_folder_name)
                                    """
                                    # Rename folder
                                    src_folder_name = os.path.join(exp_dir)
                                    dst_folder_name = os.path.join(exp_dir +
                                                                '_{}'.format(seed) +
                                                                '_{}'.format(fold) +
                                                                '_{}'.format(best_epoch) +
                                                                '_{}'.format(model_name) +
                                                                '_params_{}'.format(total_params) +
                                                                '_auc' +
                                                                '_tr_{}'.format(train_auc_value) +
                                                                '_lr_{}'.format(train_auc_lr) +
                                                                '_val_{}'.format(val_auc_value) +
                                                                '_lr_{}'.format(val_auc_lr)
                                                                )

                                    if test_dl is not None:
                                        dst_folder_name += '_test_{}'.format(test_auc_value)
                                        dst_folder_name += '_lr_{}'.format(test_auc_lr)

                                    # Results of all folds (avg and ensemble)
                                    if (fold == cv_folds - 1) and (cv_folds > 1):
                                        dst_folder_name += '_avg'

                                        if max_epochs > 0:
                                            # DL training (avg)
                                            train_auc_list = [x for x in train_auc_list if x is not None]
                                            train_auc_list_mean = round(sum(train_auc_list) / len(train_auc_list),
                                                                        nr_of_decimals)

                                        # LR training (avg)
                                        train_auc_lr_list = [x for x in train_auc_lr_list if x is not None]
                                        train_auc_lr_list_mean = round(sum(train_auc_lr_list) / len(train_auc_lr_list),
                                                                    nr_of_decimals)
                                        dst_folder_name += '_tr_{}_lr_{}'.format(train_auc_list_mean, train_auc_lr_list_mean)

                                        # Internal validation (avg) and test (ensemble)
                                        # DL model
                                        dst_folder_name += '_val'
                                        if max_epochs > 0:
                                            # Validation (avg)
                                            val_auc_list = [x for x in val_auc_list if x is not None]
                                            val_auc_list_mean = round(sum(val_auc_list) / len(val_auc_list), nr_of_decimals)

                                            # Test (ensemble)
                                            test_auc_mean = misc.compute_auc(y_pred_list=test_y_pred_ens,
                                                                            y_true_list=test_y_ens,
                                                                            auc_metric=auc_metric)
                                        dst_folder_name += '_{}'.format(val_auc_list_mean)

                                        # LR model
                                        # Validation (avg)
                                        val_auc_lr_list = [x for x in val_auc_lr_list if x is not None]
                                        val_auc_lr_list_mean = round(sum(val_auc_lr_list) / len(val_auc_lr_list),
                                                                    nr_of_decimals)
                                        dst_folder_name += '_lr_{}'.format(val_auc_lr_list_mean)
                                        # Test (ensemble)
                                        test_auc_lr_mean = misc.compute_auc(y_pred_list=test_y_pred_lr_ens,
                                                                            y_true_list=test_y_lr_ens,
                                                                            auc_metric=auc_metric)

                                        # Ensemble
                                        dst_folder_name += '_ens_{}'.format(test_auc_mean)
                                        dst_folder_name += '_lr_{}'.format(test_auc_lr_mean)

                                    shutil.move(src_folder_name, dst_folder_name)

                                    # W&B
                                    # run.name = dst_folder_name.replace(os.path.join(config.exp_root_dir), '')[1:]
                                    # # run.id = dst_folder_name.replace(os.path.join(config.exp_root_dir), '')[1:]
                                    # run.finish()



