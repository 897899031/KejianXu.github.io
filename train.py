import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yoloax import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

'''
Training your own target detection model must need to pay attention to the following points:
1, double-check before training to see if their format meets the requirements, the library requires the dataset format for the VOC format, you need to prepare the content of the input images and labels
   The input image is a .jpg image, no need to fix the size, it will be automatically resized before passing into the training.
   The grayscale image will be automatically converted to RGB image for training, no need to modify it.
   If the input image has a non-jpg suffix, you need to batch convert it to jpg and then start training.

   The label is in .xml format, and there will be information about the target to be detected in the file, and the label file corresponds to the input image file.

2, the size of the loss value is used to determine whether the convergence, it is more important to have a convergence trend, that is, the verification set loss continues to decline, if the verification set loss basically does not change, the model basically converged.
   The exact size of the loss value doesn't mean much, large and small only lies in the way the loss is calculated, not close to 0 to be good. If you want to make the loss look good, you can go directly to the corresponding loss function and divide by 10000.
   Loss values during training will be saved in the logs folder in the loss_%Y_%m_%d_%H_%M_%S folder
   
3, the trained weights file is saved in the logs folder, each training generation (Epoch) contains a number of training steps (Step), each training step (Step) for a gradient descent.
   If you just train a few Step is not saved, the concept of Epoch and Step to run through it.
'''
if __name__ == "__main__":
    #---------------------------------#
    # Cuda Whether to use Cuda
    # No GPU can be set to False
    #---------------------------------#
    Cuda = True
    #---------------------------------------------------------------------#
    # distributed is used to specify whether or not to use single multi-card distributed operation
    # Terminal commands are only supported in Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
    # DP mode is used to invoke all graphics cards by default under Windows. and DDP is not supported.
    # DP mode:
    # Set distributed = False.
    # Type CUDA_VISIBLE_DEVICES=0,1 in the terminal python train.py
    # DDP mode:
    # Set distributed = True
    # Type CUDA_VISIBLE_DEVICES=0,1 in terminal python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed = False
    #---------------------------------------------------------------------#
    # sync_bn Whether to use sync_bn, DDP mode multi-card available
    #---------------------------------------------------------------------#
    sync_bn = False
    #---------------------------------------------------------------------#
    # fp16 whether to train with mixed precision
    # Reduces video memory by about half, requires pytorch 1.7.1 or higher
    #---------------------------------------------------------------------#
    fp16 = True
    #---------------------------------------------------------------------#
    # classes_path points to the txt under model_data, which is related to your own training dataset.
    # Make sure to modify classes_path before training so that it corresponds to your own dataset.
    #---------------------------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    # See the README for the weights file, which can be downloaded via the netbook. The pre-training weights of the model are common to different datasets because the features are common.
    # Pre-training weights of the model The more important part of the model is the weights part of the backbone feature extraction network, which is used for feature extraction.
    # Pre-training weights must be used in 99% of the cases, if not, the weights of the backbone part are too random, the feature extraction is not effective, and the results of the network training are not good.
    #
    # If there is an interruption in the training process, you can set the model_path to the weights file in the logs folder, and reload the weights that have already been partially trained.
    # # Also modify the parameters of the freeze phase or unfreeze phase below to ensure the continuity of the model epoch. # # If there is an interruption in the training process, you can set model_path to the weights file in the logs folder.
    #
    # Do not load the entire model weights when model_path = ''.
    # # When model_path = '', the weights of the whole model are not loaded.
    # The weights for the whole model are used here, so they are loaded in train.py.
    # If you want the model to start training from 0, set model_path = '' and Freeze_Train = Fasle below, at this point training starts from 0 and there is no process of freezing the backbone.
    #
    # In general, the network will train poorly from 0 because the weights are too random and feature extraction is not effective, so it is very, very, very much not recommended that you start training from 0. # If you want to train from 0, you have to set the model_path = '''!
    # There are two options for training from 0:
    # 1, thanks to the powerful data enhancement capability of Mosaic data enhancement method, set UnFreeze_Epoch larger (300 and above), batch larger (16 and above), more data (more than 10,000) in the case of.
    # You can set mosaic=True and directly randomly initialize the parameters to start training, but the results obtained are still not as good as the case with pre-training. (Large datasets like COCO can do this)
    # 2, understand the imagenet dataset, first train the classification model, get the weights of the backbone part of the network, the backbone part of the classification model is common to this model, based on this training.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path = ''
    #------------------------------------------------------#
    # input_shape The input shape size, must be a multiple of 32.
    #------------------------------------------------------#
    input_shape = [640, 640]
    #------------------------------------------------------#
    # Version of YoloX used: nano, tiny, s, m, l, x
    #------------------------------------------------------#
    phi = 's'
    #------------------------------------------------------------------#
    # mosaic mosaic data augmentation.
    # mosaic_prob What is the probability of using mosaic data augmentation per STEP, default 50%.
    # mosaic_prob
    # mixup Whether to use mixup data enhancement, valid only when mosaic=True.
    # Only mixup will be applied to mosaic-enhanced images.
    # mixup_prob What is the probability to use mixup data enhancement after mosaic, default 50%.
    # The total mixup probability is mosaic_prob * mixup_prob.
    #
    # special_aug_ratio Refer to YoloX, since Mosaic generates training images that are far from the true distribution of natural images.
    # When mosaic=True, this code will turn on mosaic in the special_aug_ratio range.
    # Defaults to the first 70% epochs, 100 generations will turn on 70 generations.
    #
    # Parameters for the cosine annealing algorithm are put into the following lr_decay_type setting
    #------------------------------------------------------------------#
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7

    #----------------------------------------------------------------------------------------------------------------------------#
    # Training is divided into two phases, the freeze phase and the unfreeze phase. The freeze phase is set to meet the training needs of students with underperforming machines.
    # Freeze training requires less video memory, and in the case of a very poor video card, you can set Freeze_Epoch equal to UnFreeze_Epoch and Freeze_Train = True, at which point just freeze training is performed.
    #
    # Here we provide a number of parameter setting suggestions, you trainers according to their own needs for flexible adjustment:
    # (1) Start training from the pre-training weights of the whole model:
    # Adam:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (Freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (Not frozen)
    # SGD:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (frozen)
    # Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (not frozen)
    # Where: the UnFreeze_Epoch can be adjusted between 100-300.
    # (2) Start training from 0:
    # Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (no freeze training)
    # Where: UnFreeze_Epoch try not to be less than 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    # (3) batch_size setting:
    # As large as the graphics card can accept. Insufficient memory has nothing to do with the dataset size, please adjust the batch_size down when prompted with insufficient memory (OOM or CUDA out of memory).
    # Batch_size is affected by BatchNorm layer, batch_size is minimum 2, not 1.
    # Under normal circumstances Freeze_batch_size is recommended to be 1-2 times of Unfreeze_batch_size. It is not recommended to set the gap too large, because it is related to the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Freeze phase training parameters
    # The backbone of the model is frozen at this point and the feature extraction network is not changed
    # Occupy less memory, only fine-tune the network.
    # Init_Epoch The current training generation of the model, its value can be greater than Freeze_Epoch, as set:
    # Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    # Will skip the freeze phase and start directly from generation 60, and adjust the corresponding learning rate.
    # (to be used when breaking the freeze)
    # Freeze_Epoch model freeze training for Freeze_Epoch
    # (disabled when Freeze_Train=False)
    # Freeze_batch_size batch_size for model freeze training
    # (Expires when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16
    #------------------------------------------------------------------#
    # Training parameters for the unfreezing phase
    # The backbone of the model is not frozen at this point, the feature extraction network changes
    # The memory used is larger and all the parameters of the network are changed
    # Total number of epochs that the UnFreeze_Epoch model has been trained for
    # SGD takes longer to converge, so set a larger UnFreeze_Epoch
    # Adam can use a relatively small UnFreeze_Epoch
    # Unfreeze_batch_size batch_size of model after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    # Freeze_Train whether to do freeze training or not
    # Default to freeze the trunk training first and then unfreeze the training.
    #------------------------------------------------------------------#
    Freeze_Train = False
    
    #------------------------------------------------------------------#
    # Other training parameters: learning rate, optimizer, learning rate degradation related to
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Maximum learning rate of the Init_lr model
    # Min_lr model's minimum learning rate, defaults to 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type the type of optimizer to use, optional adam, sgd
    # Init_lr=1e-3 is recommended when using Adam optimizer.
    # Init_lr=1e-3 is recommended when using the SGD optimizer.
    # momentum The momentum parameter is used internally by the optimizer.
    # weight_decay weight_decay to prevent overfitting
    # adam will cause weight_decay error, recommend setting to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    #------------------------------------------------------------------#
    # lr_decay_type Learning rate descent method used, options are step, cos
    #------------------------------------------------------------------#
    lr_decay_type = "cos"
    #------------------------------------------------------------------#
    # save_period how many epochs to save weights once
    #------------------------------------------------------------------#
    save_period = 20
    #------------------------------------------------------------------#
    # save_dir Folder where permissions and log files are saved
    #------------------------------------------------------------------#
    save_dir = 'logs'
    #------------------------------------------------------------------#
    # eval_flag Whether eval_flag is evaluated at training time on the validation set.
    # eval_period The eval_period represents how many epochs to evaluate.
    # eval_period How many epochs should be evaluated, frequent evaluation is not recommended.
    # eval_period is the number of epochs to evaluate, frequent evaluation is not recommended.
    # The mAP obtained here will be different from the one obtained from get_map.py for two reasons:
    # (1) The mAP obtained here is the mAP of the validation set.
    # (2) The evaluation parameters here are set more conservatively to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag = True
    eval_period = 10
    #------------------------------------------------------------------#
    # num_workers is used to set whether to use multiple threads to read data.
    # When enabled, it will speed up data reading, but it will take up more memory.
    # For computers with less memory, you can set it to 2 or 0.
    #------------------------------------------------------------------#
    num_workers = 8

    #----------------------------------------------------#
    # Get image paths and tags
    #----------------------------------------------------#
    train_annotation_path = '2012_train.txt'
    val_annotation_path = '2012_val.txt'
    #------------------------------------------------------#
    # Setting up the graphics card to be used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0
        
    #----------------------------------------------------#
    # Getting classes and anchors
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    # Create yolo models
    #------------------------------------------------------#
    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        # See the README for the authority file
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        # Load based on Key of pre-training weights and Key of models
        #------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        # Show Keys with no matches
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    #----------------------#
    # Get the loss function
    #----------------------#
    yolo_loss = YOLOLoss(num_classes, fp16)
    #----------------------#
    # Record Loss
    #----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    #------------------------------------------------------------------#
    # torch 1.2 does not support amp, we recommend using torch 1.7.1 and above to use fp16 correctly.
    # So torch 1.2 shows "could not be resolved" here.
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    #----------------------------#
    # Multi-card synchronization Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            # Multi-card parallel operation
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #----------------------------#
    # Weight smoothing
    #----------------------------#
    ema = ModelEMA(model_train)
    #---------------------------#
    # Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
     
    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        # Total training generations refers to the total number of times the entire data is traversed
        # The total training steps are the total number of gradient descents
        # Each training generation contains a number of training steps, each training step performs one gradient descent.
        # Only the lowest training generation is recommended here, there is no maximum, only the unfrozen part is considered in the calculation
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small for training, please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size above %d。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data volume for this run is %d, Unfreeze_batch_size is %d, a total of %d Epochs are trained, and the total training step size is calculated as %d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training step is %d, which is less than the recommended total step %d, it is recommended to set the total generation to %d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    # Backbone features extract network features generically, freezing training can speed up training
    # It can also prevent the weights from being corrupted at the beginning of training.
    # Init_Epoch is the starting generation
    # Freeze_Epoch is the generation of freeze training.
    # UnFreeze_Epoch is the total number of training generations.
    # If you get OOM or low memory, please reduce the Batch_size.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        # Freeze a certain portion of training
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        # Set batch_size directly to Unfreeze_batch_size if you don't freeze training
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        # Determine current batch_size, adaptively adjust learning rate
        #-------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        # Select optimizer based on optimizer_type
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam' : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'  : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        # Access to formulas for decreasing learning rates
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        # Judging the length of each generation
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, please expand the dataset。")
        
        if ema:
            ema.updates     = epoch_step * Init_Epoch

        #---------------------------------------#
        # Build the dataset loader.
        #---------------------------------------#
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size = batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        #----------------------#
        # Record the map curve of eval
        #----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda,\
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        
        #---------------------------------------#
        # Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            # If the model has a frozen learning component
            # Then unfreeze and set the parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                    
                #-------------------------------------------------------------------#
                # Determine current batch_size, adaptively adjust learning rate
                #-------------------------------------------------------------------#
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                # Access to formulas for decreasing learning rates
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, please expand the dataset。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                if ema:
                    ema.updates = epoch_step * epoch
                    
                gen = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
