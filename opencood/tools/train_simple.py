# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import time

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
# from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.clip_grad import clip_grad_norm_

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import glob
from icecream import ic
import datetime
from opencood.utils import common_utils
from tqdm import tqdm
from tqdm.contrib import tenumerate
from collections import defaultdict

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    # criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    log_file = os.path.join(saved_path, ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info(f"===log dir: {saved_path}===")

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(opt).items():
        logger.info('{:16} {}'.format(key, val))

    logger.info(model)
    num_total_params = sum([x.numel() for x in model.parameters()])
    logger.info(f'Total number of parameters: {num_total_params}')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(os.path.join(saved_path, 'tensorboard'))

    start_time_epoch = time.time()
    batch_len = len(train_loader)
    logger.info('Training start')
    epoches = hypes['train_params']['epoches']
    max_epoch = max(epoches, init_epoch)

    logger.info(f"===total epoches is {epoches}===")
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max_epoch):
        for param_group in optimizer.param_groups:
            logger.info('learning rate %f' % param_group["lr"])
            cur_lr = param_group["lr"]
        pbar2 = tqdm(total=batch_len, leave=True, colour='#DDA0DD') # 进度条
        start_time_batch = time.time()
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0: # 没有gt
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch # type: ignore
            ouput_dict = model(batch_data['ego']) # type: ignore
            final_loss, tb_dict = model(batch_data['ego']) # type: ignore
            final_loss = final_loss.mean()
            # final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            # criterion.logging(epoch, i, len(train_loader), writer)

            # if supervise_single_flag:
            #     final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") # type: ignore
            #     criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            sample_idx = epoch*batch_len + i
            pbar2.set_description("[epoch %d][%d/%d]%s, || Loss: %.4f" %(epoch, i + 1, batch_len, '', final_loss))
            pbar2.update()
            writer.add_scalar('train/loss', final_loss, sample_idx)
            for key, val in tb_dict.items():
                writer.add_scalar('train/' + key, val, sample_idx)

            # BUG 调试
            # 对单独的 loss_giou 计算梯度
            # optimizer.zero_grad()
            # tb_dict["loss_bbox_debug"].backward(retain_graph=True)
            # bbox_grad_norm = sum(torch.norm(param.grad) for param in model.parameters() if param.grad is not None)
            # print(f"Gradient norm for BBox loss: {bbox_grad_norm:.4f}")
            # # 对单独的 loss_giou 计算梯度
            # optimizer.zero_grad()
            # tb_dict["loss_giou_debug"].backward(retain_graph=True)
            # giou_grad_norm = sum(torch.norm(param.grad) for param in model.parameters() if param.grad is not None)
            # print(f"Gradient norm for GIoU loss: {giou_grad_norm:.4f}")

            # # 对单独的 loss_rad 计算梯度
            # optimizer.zero_grad()
            # tb_dict["loss_rad_debug"].backward(retain_graph=True)
            # rad_grad_norm = sum(torch.norm(param.grad) for param in model.parameters() if param.grad is not None)
            # print(f"Gradient norm for Rad loss: {rad_grad_norm:.4f}")
            # optimizer.zero_grad()

            # back-propagation
            final_loss.backward()

            # print("Gradient norm for classification head:", torch.norm(model.dense_head.transformer.proposal_head.class_embed[0].layers[-1].weight.grad))
            # print("Gradient norm for bbox head:", torch.norm(model.dense_head.transformer.proposal_head.bbox_embed[0].layers[-1].weight.grad))
            # print("Gradient norm for bbox head:", torch.norm(model.bbox_embed[0].weight.grad))

            total_norm = clip_grad_norm_(model.parameters(), 10) # 梯度剪裁，防止梯度爆炸
            optimizer.step()

            # torch.cuda.empty_cache()
        writer.add_scalar('meta_data/learning_rate', cur_lr, epoch) # 学习率train一轮后记录一次
        end_time = time.time()
        # second_each_iter = pbar2.format_dict['elapsed'] / max(batch_len, 1.0)
        epoch_cost = (end_time - start_time_batch)/60
        all_cost = (end_time - start_time_epoch)/60
        remain_cost = max_epoch * epoch_cost - all_cost
        # gpu_info = os.popen('gpustat').read().encode('utf-8').decode('utf-8')
        # logger.info(gpu_info)
        logger.info('### %d th epoch trained, start validation! Time_cost(epoch): %.2f, Time cost(all): %.2f, Remain time: %.2f  ###' % (epoch, epoch_cost, all_cost, remain_cost))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            
            many_loss_dict = defaultdict(lambda: 0)
            with torch.no_grad():
                for i, batch_data in tenumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch # type: ignore
                    final_loss, tb_dict = model(batch_data['ego']) # type: ignore

                    # final_loss = criterion(ouput_dict,
                    #                        batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
                    for key, val in tb_dict.items():
                        many_loss_dict[key] += val

            valid_ave_loss = statistics.mean(valid_ave_loss) # 求平均损失
            logger.info('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('validate/loss', valid_ave_loss, epoch)
            for key, val in many_loss_dict.items():
                writer.add_scalar('validate/' + key, val, epoch)
            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step()

        opencood_train_dataset.reinitialize()

    logger.info('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = False    
    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
        import numpy as np
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    if run_test:
        fusion_method = opt.fusion_method
        if 'noise_setting' in hypes and hypes['noise_setting']['add_noise']:
            cmd = f"python opencood/tools/inference_w_noise.py --model_dir {saved_path} --fusion_method {fusion_method}"
        else:
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        logger.info(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
