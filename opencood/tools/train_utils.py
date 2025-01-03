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
import torch.nn as nn
import numpy as np
from functools import partial
from .fastai_optim import OptimWrapper

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
    lr = method_dict.get('initial_lr', 0.001)  # initial learning rate
    initial_lr = method_dict.get('initial_lr', lr)  # 如果没有指定 initial_lr，则使用 lr 作为初始学习率
    # 下面的都是用在oneCycle策略中的
    # max_lr = method_dict.get('max_lr', lr)
    # min_lr = method_dict.get('min_lr', lr)
    # max_momentum = method_dict.get('max_momentum', 0.95)  
    # base_momentum = method_dict.get('base_momentum', 0.85)

    if not optimizer_method and method_dict['core_method'] != 'adam_onecycle':
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict and method_dict['core_method'] != 'adam_onecycle':
        optimizer = optimizer_method(model.parameters(),
                                lr=lr,
                                **method_dict['args'])
    elif method_dict['core_method'] == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=method_dict['args']['weight_decay'], true_wd=True, bn_wd=True
        )
    else:
        optimizer = optimizer_method(model.parameters(),
                                lr=lr)
    # 确保每个 param_group 都有参数
    # for param_group in optimizer.param_groups:
    #     param_group['initial_lr'] = initial_lr
        # param_group['max_lr'] = max_lr  # 添加 max_lr
        # param_group['min_lr'] = min_lr
        # param_group['max_momentum'] = max_momentum  # 添加 max_momentum
        # param_group['base_momentum'] = base_momentum
    return optimizer


def setup_lr_schedular(hypes, optimizer, init_epoch=None, total_iters_each_epoch=None):
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
    elif lr_schedule_config['core_method'] == 'onecycle':
        total_steps = total_iters_each_epoch * hypes['train_params']['epoches'] # 总训练样本数
        scheduler = OneCycle(
            optimizer, total_steps, lr_schedule_config['max_lr'], lr_schedule_config['moms'], lr_schedule_config['div_factor'], lr_schedule_config['pct_start']
        )
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
        
class LRSchedulerStep(object):
    def __init__(self, fai_optimizer: OptimWrapper, total_step, lr_phases,
                 mom_phases):
        # if not isinstance(fai_optimizer, OptimWrapper):
        #     raise TypeError('{} is not a fastai OptimWrapper'.format(
        #         type(fai_optimizer).__name__))
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        # self.lr_phases 加入了两个元素，表示两个阶段，每个阶段有(开始iters, 结束iters, lambda函数)
        for i, (start, lambda_func) in enumerate(lr_phases): # 学习率的每个阶段
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append((int(start * total_step), int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0 # 必须是从头开始
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0

    def step(self, step):
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start):
        '''
        fai_optimizer: 优化器对象
        total_step: 训练总样本数 = epochs * iters_per_epoch
        lr_max: 最大学习率 0.002
        moms: [ 0.95, 0.85 ]
        div_factor: 10 初始学习率和最高学习率的比例
        pct_start: 0.4 上升阶段的比例
        '''
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start) # 上升阶段占据0.4
        a2 = total_step - a1
        low_lr = self.lr_max / self.div_factor # 这个应该是用于作为初始学习率
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)