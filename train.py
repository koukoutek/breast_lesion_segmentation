import torch
import os
import random
import traceback
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from monai.data import DataLoader, CacheDataset, Dataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.data.utils import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, UNETR
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import (ToTensord, Compose, LoadImaged, ToTensord, RandCropByPosNegLabeld, Transposed, RandCropByLabelClassesd, CropForegroundd,
                              EnsureType, Compose, AsDiscrete, RandSpatialCropSamplesd, SpatialPadd, 
                              Resized, RandFlipd, RandShiftIntensityd, RandAdjustContrastd)
from utils import *
from pathlib import Path


# from tqdm import tqdm # can be added if not running in the background

def save_checkpoint(model_state_dict, 
                    optimizer_seg_state_dict, 
                    save_path=None):
    """Save checkpoint while training the model

    Args:
        model_state_dict (dict): Dictionary containing model state i.e. weights and biases
            Required: True
        optimizer_state_dict (dict): Dictionary containing optimizer state for the segmentation part i.e. gradients
            Required: True
        save_path (str): Path to save the checkpoint
            Required: False     Default: None  
    Returns:
        -
    """
    torch.save({'model_state_dict': model_state_dict,
                'optimizer_seg_state_dict': optimizer_seg_state_dict,
                }, save_path)
    
def key_error_raiser(ex): raise Exception(ex)

def train(config, log_path, logger):
    import pandas as pd

    df = pd.read_csv(r'D:\Users\kkouko0\DATASETS\INBreast\INbreast Release 1.0\lesion_on_cases.csv')
    df = df[df['lesion'] == 1]
    df['case'] = df['case'].astype(str)

    cases = sorted([f for f in Path(config['data_dir']).joinpath('AllDICOMs').glob('*') if str(f).endswith('.dcm')])
    masks = sorted([f for f in Path(config['data_dir']).joinpath('AllMasks').glob('*')])

    image_files = sorted([f for f in cases if f.stem.split('_')[0] in [k.stem for k in masks] and f.stem.split('_')[0] in df['case'].tolist()])
    mask_files = sorted([f for f in masks if f.stem in df['case'].tolist()])

    train_transforms_config = config['train_transforms']
    eval_transforms_config = config['eval_transforms']
    train_transforms = Compose([
                            # # Load and preprocess
                            LoadImaged(keys=train_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=train_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            LoadImaged(keys=train_transforms_config['LoadImaged_seg']['keys'], ensure_channel_first=train_transforms_config['LoadImaged_seg']['ensure_channel_first']),
                            Transposed(keys=train_transforms_config['Transposed']['keys'], indices=train_transforms_config['Transposed']['indices']),
                            WindowindINBreastImageBasedOnPercentiled(keys=train_transforms_config['WindowindINBreastImageBasedOnPercentiled']['keys']),
                            
                            # # Resize
                            Resized(keys=train_transforms_config['Resized_im']['keys'], spatial_size=train_transforms_config['Resized_im']['spatial_size']),
                            Resized(keys=train_transforms_config['Resized_seg']['keys'], spatial_size=train_transforms_config['Resized_seg']['spatial_size'], 
                                    mode=train_transforms_config['Resized_seg']['mode']),

                            # # Crop foreground
                            CropForegroundd(keys=train_transforms_config['CropForegroundd']['keys'], source_key=train_transforms_config['CropForegroundd']['source_key']),
                            SpatialPadd(keys=train_transforms_config['SpatialPadd']['keys'], 
                                        spatial_size=train_transforms_config['SpatialPadd']['spatial_size']),
                            ConvertINBreastLesionToMultiChannelMaskd(keys=train_transforms_config['ConvertINBreastLesionToMultiChannelMaskd']['keys']),                       
                            RandCropByLabelClassesd(keys=train_transforms_config['RandCropByLabelClassesd']['keys'], 
                                                   label_key=train_transforms_config['RandCropByLabelClassesd']['label_key'], 
                                                   spatial_size=train_transforms_config['RandCropByLabelClassesd']['spatial_size'], 
                                                   num_classes=train_transforms_config['RandCropByLabelClassesd']['num_classes'],
                                                   ratios=train_transforms_config['RandCropByLabelClassesd']['ratios'],
                                                   num_samples=train_transforms_config['RandCropByLabelClassesd']['num_samples']),

                            # # Augmentations
                            RandFlipd(keys=train_transforms_config['RandFlipd_x']['keys'], 
                                      spatial_axis=train_transforms_config['RandFlipd_x']['spatial_axis'],
                                      prob=train_transforms_config['RandFlipd_x']['prob']),
                            RandFlipd(keys=train_transforms_config['RandFlipd_y']['keys'], 
                                      spatial_axis=train_transforms_config['RandFlipd_y']['spatial_axis'],
                                      prob=train_transforms_config['RandFlipd_y']['prob']),
                            RandShiftIntensityd(keys=train_transforms_config['RandShiftIntensityd']['keys'],
                                                offsets=train_transforms_config['RandShiftIntensityd']['offsets'],
                                                prob=train_transforms_config['RandShiftIntensityd']['prob']),
                            RandAdjustContrastd(keys=train_transforms_config['RandAdjustContrastd']['keys'],
                                                gamma=train_transforms_config['RandAdjustContrastd']['gamma'],
                                                prob=train_transforms_config['RandAdjustContrastd']['prob']),
                            
                            # # Return to tensor
                            ToTensord(keys=train_transforms_config['ToTensord']['keys'])
                        ])

    val_transforms = Compose([
                            # # Load and preprocess
                            LoadImaged(keys=eval_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=eval_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            LoadImaged(keys=eval_transforms_config['LoadImaged_seg']['keys'], ensure_channel_first=eval_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            Transposed(keys=eval_transforms_config['Transposed']['keys'], indices=eval_transforms_config['Transposed']['indices']),
                            WindowindINBreastImageBasedOnPercentiled(keys=eval_transforms_config['WindowindINBreastImageBasedOnPercentiled']['keys']),
                            
                            # # Resize
                            Resized(keys=eval_transforms_config['Resized_im']['keys'], spatial_size=eval_transforms_config['Resized_im']['spatial_size']),
                            Resized(keys=eval_transforms_config['Resized_seg']['keys'], spatial_size=eval_transforms_config['Resized_seg']['spatial_size'], 
                                    mode=eval_transforms_config['Resized_seg']['mode']),

                            # # Crop foreground
                            CropForegroundd(keys=eval_transforms_config['CropForegroundd']['keys'], source_key=eval_transforms_config['CropForegroundd']['source_key']),
                            SpatialPadd(keys=eval_transforms_config['SpatialPadd']['keys'], 
                                        spatial_size=eval_transforms_config['SpatialPadd']['spatial_size']),
                            ConvertINBreastLesionToMultiChannelMaskd(keys=eval_transforms_config['ConvertINBreastLesionToMultiChannelMaskd']['keys']),   
                            RandCropByLabelClassesd(keys=eval_transforms_config['RandCropByLabelClassesd']['keys'], 
                                                   label_key=eval_transforms_config['RandCropByLabelClassesd']['label_key'], 
                                                   spatial_size=eval_transforms_config['RandCropByLabelClassesd']['spatial_size'], 
                                                   num_classes=eval_transforms_config['RandCropByLabelClassesd']['num_classes'],
                                                   ratios=eval_transforms_config['RandCropByLabelClassesd']['ratios'],
                                                   num_samples=eval_transforms_config['RandCropByLabelClassesd']['num_samples']),

                            # # Return to tensor
                            ToTensord(keys=eval_transforms_config['ToTensord']['keys'])
                        ])

    datadict = [{"image": im, "mask": mask} for im, mask in zip(image_files, mask_files)]
    # datadict = datadict[:100]
    
    cross_val_split = config['cross_val_split'] if 'cross_val_split' in config.keys() else key_error_raiser("Cross validation split not defined in config.")
    
    random.shuffle(datadict)
    
    test_dict = datadict[:int(len(datadict) * cross_val_split[2])]
    val_dict = datadict[int(len(datadict) * cross_val_split[1]):int(len(datadict) * cross_val_split[2])]
    train_dict = datadict[int(len(datadict) * cross_val_split[1])+int(len(datadict) * cross_val_split[2]):]

    # val_dict = datadict[int(len(datadict) * cross_val_split):]
    train_dict = train_dict
    val_dict = val_dict
    logger.info('Train/Val split {} , {}'. format(len(train_dict), len(val_dict)))

    # define dataset
    if config['dataset'] == 'Dataset':
        train_dataset = Dataset(data=train_dict, transform=train_transforms)
        val_dataset = Dataset(data=val_dict, transform=val_transforms)
    elif config['dataset'] == 'CacheDataset':
        train_dataset = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=config['cache_rate'])
        val_dataset = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=config['cache_rate'])

    # train_size = len(train_dataset)
    val_size = len(val_dataset)

    # initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'] , 
                              shuffle=config['dataloader']['shuffle'], 
                              num_workers=config['dataloader']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=1, 
                            shuffle=config['dataloader']['shuffle'],
                            num_workers=config['dataloader']['num_workers'])

    # initialize model
    if config['model']['name'] == 'UNETR': 
        model = UNETR(in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'], img_size=config['model']['img_size'], 
                      feature_size=config['model']['feature_size'], hidden_size=config['model']['hidden_size'], mlp_dim=config['model']['mlp_dim'], 
                      num_heads=config['model']['num_heads'], pos_embed=config['model']['pos_embed'], norm_name=config['model']['norm_name'], 
                      conv_block=config['model']['conv_block'], res_block=config['model']['res_block'], dropout_rate=config['model']['dropout_rate'])
    elif config['model']['name'] == 'UNet':
        model = UNet(spatial_dims=config['model']['spatial_dims'], in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'],
                     kernel_size=config['model']['kernel_size'], up_kernel_size=config['model']['up_kernel_size'], channels=config['model']['channels'],
                     strides=config['model']['strides'], norm=config['model']['norm'], dropout=config['model']['dropout'], 
                     num_res_units=config['model']['num_res_units'])
    else: 
        raise Exception("No model has been defined in the config file")
    
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model) # use multiple GPUs
    model.to(device=torch.device(config['device']))
        
    # initialize optimizer
    if config['optimizer']['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['learning_rate'], betas=config['optimizer']['betas'], 
                                     weight_decay=config['optimizer']['weight_decay']) 
    else: 
        raise Exception("No optimizer has been defined in the config file")
    logger.info('Training with optimizer {} '.format(optimizer))

    # initialize scheduler
    if config['scheduler']['name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=config['scheduler']['T_0'], eta_min=config['scheduler']['eta_min'])
        logger.info('Scheduler {}'.format(scheduler))
    else:
        logger.info('No scheduler for the learning rate has been defined')

    # initialize loss
    if config['loss']['name'] == 'DiceLoss':
        loss = DiceLoss(softmax=config['loss']['softmax'], include_background=config['loss']['include_background'])
    elif config['loss']['name'] == 'DiceCELoss':
        loss = DiceCELoss(softmax=config['loss']['softmax'], include_background=config['loss']['include_background'])
    else:
        raise Exception("No loss has been defined in the config file")
    logger.info('Loss function to minimize {}'.format(loss))

    if config['save_model_when']:
        metric_threshold = config['save_model_when']
    else:
        metric_threshold = 0.95

    writer = SummaryWriter()

    losses = []
    val_losses = []

    post_pred_transforms = config['post_pred_transforms']
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=post_pred_transforms['AsDiscrete']['argmax'], 
                                                  to_onehot=post_pred_transforms['AsDiscrete']['to_onehot'])])
    post_label = Compose([EnsureType()])

    if config['metric']['name'] == 'DiceMetric':
        train_metric = DiceMetric(include_background=config['metric']['include_background'], reduction=config['metric']['reduction'])
        val_metric =  DiceMetric(include_background=config['metric']['include_background'], reduction=config['metric']['reduction'])
    else:
        raise Exception("No metric has been defined in the config file")
    logger.info('Metric {}'.format(train_metric))

    for epoch in range(config['epochs']):
        model.train()

        for batch, train_data in enumerate(train_loader, 1):
            image, segmentation = train_data['image'].float().to(device=torch.device(config['device'])), train_data['segmentation'].float().to(device=torch.device(config['device']))

            try:
                optimizer.zero_grad()
                out = model(image)

                loss_s = loss(out, segmentation)
                loss_s.backward()

                _outputs = [post_pred(i) for i in decollate_batch(out)]
                _labels = [post_label(i) for i in decollate_batch(segmentation)]

                optimizer.step()
                train_metric(y_pred=_outputs, y=_labels)

            except Exception as e:
                print('Caught the following exception {}'.format(traceback.format_exc()))
            losses.append(loss_s.item())
        metric = train_metric.aggregate().item()
        scheduler.step()

        if epoch > 500 and epoch % 50 == 0 and metric > 0.9:
            plot_2d_or_3d_image(data=image, step=0, writer=writer, frame_dim=-1, tag=f'image at epoch: {epoch}')
            plot_2d_or_3d_image(data=segmentation, step=0, writer=writer, frame_dim=-1, tag=f'label at epoch: {epoch}')
            plot_2d_or_3d_image(data=out, step=0, writer=writer, frame_dim=-1, tag=f'model output at epoch: {epoch}')

        writer.add_scalar(tag='Loss/train', scalar_value=losses[-1], global_step=epoch)
        logger.info(f'Epoch {epoch} of {config["epochs"]} with Train loss {losses[-1]}')
        logger.info(f'Epoch {epoch} of {config["epochs"]} with Train metric {metric}')
        logger.info(f'-------------- Finished epoch {epoch} -------------')
        train_metric.reset()

        if epoch % config['val_interval'] == 0:
            with torch.no_grad():
                # evaluate model
                model.eval()

                for _, val_data in enumerate(val_loader, 1):
                    val_image, val_segm = val_data['image'].float().to(device=torch.device(config['device'])), val_data['segmentation'].float().to(device=torch.device(config['device']))

                    try:
                        val_out = sliding_window_inference(inputs=val_image, roi_size=[1024, 1024], sw_batch_size=12, predictor=model, overlap=0.5)

                        loss_s = loss(val_out, val_segm)

                        val_outputs = [post_pred(i) for i in decollate_batch(val_out)]
                        val_labels = [post_label(i) for i in decollate_batch(val_segm)]

                        val_metric(val_outputs, val_labels)
                    except Exception as e:
                        print(f'Exception caught while validating in {traceback.format_exc()}. Aborting...')
                    # record loss
                    val_losses.append(loss_s.item())
                metric = val_metric.aggregate().item()

                writer.add_scalar(tag='Loss/eval', scalar_value=val_losses[-1], global_step=epoch)
                logger.info(f'Eval loss {val_losses[-1]}')
                logger.info(f'Eval metric {metric}')
                logger.info(f'-------------- Finished epoch {epoch} -------------') 
                val_metric.reset()

                # save models
                if metric > metric_threshold:
                    if not os.path.exists(log_path.joinpath(config['logs']).joinpath('models')):
                        os.makedirs(log_path.joinpath(config['logs']).joinpath('models'))
                    save_checkpoint(model_state_dict=model.state_dict(), optimizer_seg_state_dict=optimizer.state_dict(), 
                                    save_path=log_path.joinpath(config['logs']).joinpath('models/model{}.tar'.format(epoch)))

    return model  