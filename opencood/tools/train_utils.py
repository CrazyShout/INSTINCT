# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        model.load_state_dict(torch.load(file_list[0] , map_location='cpu'), strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(
            os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch), map_location='cpu'), strict=False)

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    dataset = hypes.get('dataset', 'dairv2x')
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')
    current_path = os.path.join(current_path, dataset)

    full_path = os.path.join(current_path, folder_name)
    full_path = os.path.abspath(full_path)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    # 获取 initial_lr 和 lr 的值
    lr = method_dict['lr']  # initial learning rate
    initial_lr = method_dict.get('initial_lr', lr)  # 如果没有指定 initial_lr，则使用 lr 作为初始学习率
    # 下面的都是用在oneCycle策略中的
    # max_lr = method_dict.get('max_lr', lr)
    # min_lr = method_dict.get('min_lr', lr)
    # max_momentum = method_dict.get('max_momentum', 0.95)  
    # base_momentum = method_dict.get('base_momentum', 0.85)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        optimizer = optimizer_method(model.parameters(),
                                lr=lr,
                                **method_dict['args'])
    else:
        optimizer = optimizer_method(model.parameters(),
                                lr=lr)
    # 确保每个 param_group 都有参数
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = initial_lr
        # param_group['max_lr'] = max_lr  # 添加 max_lr
        # param_group['min_lr'] = min_lr
        # param_group['max_momentum'] = max_momentum  # 添加 max_momentum
        # param_group['base_momentum'] = base_momentum
    return optimizer


def setup_lr_schedular(hypes, optimizer, init_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'fade':
        total_epoches = hypes['train_params']['epoches']
        max_lr = lr_schedule_config['max_lr']
        fade_in_ratio = lr_schedule_config['fade_in_ratio']
        fade_out_ratio = lr_schedule_config['fade_out_ratio']
        scheduler = FadeScheduler(optimizer, max_lr, total_epoches, fade_in_ratio, fade_out_ratio, last_epoch)    
    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)

from torch.optim.lr_scheduler import _LRScheduler

class FadeScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_epochs, fade_in_ratio=0.3, fade_out_ratio=0.7, last_epoch=-1, end_lr=None):
        """
        Args:
            optimizer: 优化器
            max_lr: 最大学习率
            total_epochs: 总训练轮数
            fade_in_ratio: 学习率增长阶段占总轮数的比例
            fade_out_ratio: 学习率衰减阶段开始的时间点占总轮数的比例
            last_epoch: 上一轮的epoch索引
            end_lr: 最终学习率，如果不指定则为初始学习率的1/10
        """
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.fade_in_epochs = int(total_epochs * fade_in_ratio)
        self.fade_out_epochs = int(total_epochs * fade_out_ratio)
        self.start_lr = optimizer.defaults['lr']
        print(self.start_lr)
        self.end_lr = end_lr if end_lr is not None else self.start_lr / 10
        self.last_epoch = last_epoch
        self.base_lrs = [optimizer.defaults['lr']]
        super(FadeScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 基于当前epoch计算学习率
        current_epoch = self.last_epoch + 1
        
        if current_epoch <= self.fade_in_epochs:
            # 增长阶段：从start_lr线性增长到max_lr
            scale = current_epoch / self.fade_in_epochs
            return [self.start_lr + scale * (self.max_lr - self.start_lr) for base_lr in self.base_lrs]
            
        elif current_epoch <= self.fade_in_epochs + self.fade_out_epochs:
            # 衰减阶段：从max_lr线性衰减到end_lr
            scale = (current_epoch - self.fade_in_epochs) / self.fade_out_epochs
            return [self.max_lr - scale * (self.max_lr - self.end_lr) for base_lr in self.base_lrs]
            
        else:
            # 保持阶段：维持在end_lr
            return [self.end_lr for base_lr in self.base_lrs]