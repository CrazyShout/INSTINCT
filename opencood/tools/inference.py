# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from tqdm.contrib import tenumerate

torch.multiprocessing.set_sharing_strategy('file_system')

def apply_dropout(m):
    from torch import nn
    if type(m) == nn.Dropout2d or type(m) == nn.Dropout:
        m.train()

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    # if 'heter' in hypes:
    #     x_min, x_max = -140.8, 140.8
    #     y_min, y_max = -40, 40
    #     opt.note += f"_{x_max}_{y_max}"
    #     hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
    #     hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
    #     hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

    #     new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
    #                         x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
    #     hypes['preprocess']['cav_lidar_range'] =  new_cav_range
    #     hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
    #     hypes['postprocess']['gt_range'] = new_cav_range
    #     hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
    #     if 'camera_mask_args' in hypes['model']['args']:
    #         hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

    #     # reload anchor
    #     yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    #     for name, func in yaml_utils_lib.__dict__.items():
    #         if name == hypes["yaml_parser"]:
    #             parser_func = func
    #     hypes = parser_func(hypes)
        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("my_opv2v" in hypes['test_dir'] or "my_v2xset" in hypes['test_dir']) else False
    # left_hand = True

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if opt.fusion_method == 'no_w_uncertainty':
        print("===activate dropout in inference stage!===")
        model.apply(apply_dropout)
    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'uncertainty_x_fp':[], 'uncertainty_y_fp':[], 'uncertainty_a_fp':[], 'uncertainty_x_tp':[], 'uncertainty_y_tp':[], 'uncertainty_a_tp':[], 'uncertainty_ale_cls_fp':[], 'uncertainty_ale_cls_tp':[], 'uncertainty_epi_cls_fp':[], 'uncertainty_epi_cls_tp':[], 'uncertainty_epi_reg_fp':[], 'uncertainty_epi_reg_tp':[], 'score_tp':[], 'distance_tp':[], 'distance_tp_ab':[], 'distance_all':[]},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'uncertainty_x_fp':[], 'uncertainty_y_fp':[], 'uncertainty_a_fp':[], 'uncertainty_x_tp':[], 'uncertainty_y_tp':[], 'uncertainty_a_tp':[], 'uncertainty_ale_cls_fp':[], 'uncertainty_ale_cls_tp':[], 'uncertainty_epi_cls_fp':[], 'uncertainty_epi_cls_tp':[], 'uncertainty_epi_reg_fp':[], 'uncertainty_epi_reg_tp':[], 'score_tp':[], 'distance_tp':[], 'distance_tp_ab':[], 'distance_all':[]},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'uncertainty_x_fp':[], 'uncertainty_y_fp':[], 'uncertainty_a_fp':[], 'uncertainty_x_tp':[], 'uncertainty_y_tp':[], 'uncertainty_a_tp':[], 'uncertainty_ale_cls_fp':[], 'uncertainty_ale_cls_tp':[], 'uncertainty_epi_cls_fp':[], 'uncertainty_epi_cls_tp':[], 'uncertainty_epi_reg_fp':[], 'uncertainty_epi_reg_tp':[], 'score_tp':[], 'distance_tp':[], 'distance_tp_ab':[], 'distance_all':[]}}

    
    infer_info = opt.fusion_method + opt.note

    print(f"==={infer_info}===")
    for i, batch_data in tenumerate(data_loader):
        # print(f"{infer_info}_{i}") #
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            if opt.fusion_method == 'no_w_uncertainty':
                cls_noise = infer_result['cls_noise']
                uncertainty_tensor = infer_result['uncertainty_tensor']
                unc_epi_cls = infer_result['unc_epi_cls']
                unc_epi_reg = infer_result['unc_epi_reg']
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3,
                                        cls_noise,
                                        uncertainty_tensor,
                                        unc_epi_cls,
                                        unc_epi_reg)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5,
                                        cls_noise,
                                        uncertainty_tensor,
                                        unc_epi_cls,
                                        unc_epi_reg)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7,
                                        cls_noise,
                                        uncertainty_tensor,
                                        unc_epi_cls,
                                        unc_epi_reg)
            else:
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)
                
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                """
                If you want 3D visualization, uncomment lines below
                """
                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    if opt.fusion_method == 'no_w_uncertainty':
        uncertainty_save_path_root = os.path.join(opt.model_dir, f'unc_{infer_info}')
        if not os.path.exists(uncertainty_save_path_root):
            os.makedirs(uncertainty_save_path_root)
        from datetime import datetime
        current_time = datetime.now()
        folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
        unc_log_file = os.path.join(uncertainty_save_path_root, f'output_{folder_name}.txt')
        print(f'Logging to: {unc_log_file}')
        import logging
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        #                                                                                             logging.FileHandler(unc_log_file),
        #                                                                                             logging.StreamHandler()
        #                                                                                             ])
        # logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(unc_log_file), logging.StreamHandler()])
        # 配置 logging 模块
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.INFO)

        # 创建文件处理器和控制台处理器
        file_handler = logging.FileHandler(unc_log_file)
        console_handler = logging.StreamHandler()

        # 设置日志格式
        formatter = logging.Formatter('%(message)s')  # 仅包含消息内容
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        for iou_thresh, item in result_stat.items():
            unc_x_fp_mean = np.mean(item['uncertainty_x_fp'])
            unc_y_fp_mean = np.mean(item['uncertainty_y_fp'])
            unc_a_fp_mean = np.mean(item['uncertainty_a_fp'])
            unc_x_tp_mean = np.mean(item['uncertainty_x_tp'])
            unc_y_tp_mean = np.mean(item['uncertainty_y_tp'])
            unc_a_tp_mean = np.mean(item['uncertainty_a_tp'])
            # TODO：这里需要更改，因为在MC Dropout已经将三者合一
            reg_fp = item['uncertainty_x_fp']
            reg_tp = item['uncertainty_x_tp']
            # reg_fp =  [x + y + a for x, y, a in zip(item['uncertainty_x_fp'], item['uncertainty_y_fp'], item['uncertainty_a_fp'])]
            # reg_tp =  [x + y + a for x, y, a in zip(item['uncertainty_x_tp'], item['uncertainty_y_tp'], item['uncertainty_a_tp'])]
            reg_fp_mean = np.mean(reg_fp)
            reg_tp_mean = np.mean(reg_tp)
            ale_reg_all = reg_fp + reg_tp
            ale_reg_all_mean = np.mean(ale_reg_all)
            ale_reg_all_std = np.std(ale_reg_all)

            unc_ale_cls_fp_mean = np.mean(item['uncertainty_ale_cls_fp'])
            unc_ale_cls_tp_mean = np.mean(item['uncertainty_ale_cls_tp'])
            unc_ale_cls_tp_std = np.std(item['uncertainty_ale_cls_tp'])

            unc_epi_cls_fp_mean = np.mean(item['uncertainty_epi_cls_fp'])
            unc_epi_cls_tp_mean = np.mean(item['uncertainty_epi_cls_tp'])
            unc_epi_reg_fp_mean = np.mean(item['uncertainty_epi_reg_fp'])
            unc_epi_reg_tp_mean = np.mean(item['uncertainty_epi_reg_tp'])
            unc_epi_reg_all = item['uncertainty_epi_reg_fp'] + item['uncertainty_epi_reg_tp']
            unc_epi_reg_all_mean = np.mean(unc_epi_reg_all)
            unc_epi_reg_all_std = np.std(unc_epi_reg_all)

            unc_epi_cls_tp_std = np.std(item['uncertainty_epi_cls_tp'])

            score_tp_mean = np.mean(item['score_tp'])
            score_tp_std = np.std(item['score_tp'])
            distance_tp = np.mean(item['distance_tp'])
            distance_tp_ab = np.mean(item['distance_tp_ab'])
            distance_all_mean = np.mean(item['distance_all'])
            distance_all_std = np.std(item['distance_all'])
            # condition = (remain_past1_scores > 0.6653) & (u_reg_1[remain_past1_ids] <= -0.2271) & (u_cls_1[remain_past1_ids] >= 0.7891)
            # condition = (item['uncertainty_x_fp'] <= 0.2271) & (item['uncertainty_ale_cls_fp'] >= 0.7891)


            logger.info(f'---------------IOU@{iou_thresh}---------------')
            logger.info('The Average x fp Uncertainy is %.4f, The Average x tp Uncertainty is %.4f' % (unc_x_fp_mean, unc_x_tp_mean))
            logger.info('The Average y fp Uncertainty is %.4f, The Average y tp Uncertainty is %.4f' % (unc_y_fp_mean, unc_y_tp_mean))
            logger.info('The Average a fp Uncertainty is %.4f, The Average a tp Uncertainty is %.4f' % (unc_a_fp_mean, unc_a_tp_mean))
            logger.info('The Average aleatoric(data) reg fp Uncertainty is %.4f, The Average aleatoric(data) reg tp Uncertainty is %.4f' % (reg_fp_mean, reg_tp_mean))
            logger.info('The Average aleatoric(data) cls fp Uncertainty is %.4f, The Average aleatoric(data) cls tp Uncertainty is %.4f' % (unc_ale_cls_fp_mean, unc_ale_cls_tp_mean))
            logger.info('The Average epistemic(model) cls fp Uncertainty is %.4f, The Average epistemic(model) cls tp Uncertainty is %.4f' % (unc_epi_cls_fp_mean, unc_epi_cls_tp_mean))
            logger.info('The Average epistemic(model) reg fp Uncertainty is %.4f, The Average epistemic(model) reg tp Uncertainty is %.4f' % (unc_epi_reg_fp_mean, unc_epi_reg_tp_mean))
            
            # logger.info('The Average x fp Uncertainy is %.4f, '
            #     'The Average x tp Uncertainty is %.4f, \n'
            #     'The Average y fp Uncertainty is %.4f, '
            #     'The Average y tp Uncertainty is %.4f, \n'
            #     'The Average a fp Uncertainty is %.4f, '
            #     'The Average a tp Uncertainty is %.4f, \n'
            #     'The Average reg fp Uncertainty is %.4f, '
            #     'The Average reg tp Uncertainty is %.4f, \n'
            #     'The Average aleatoric cls fp Uncertainty is %.4f, '
            #     'The Average aleatoric cls tp Uncertainty is %.4f, \n'
            #     'The Average epistemic cls fp Uncertainty is %.4f, '
            #     'The Average epistemic cls tp Uncertainty is %.4f, \n'
            #     'The Average epistemic reg fp Uncertainty is %.4f, '
            #     'The Average epistemic reg tp Uncertainty is %.4f' % (unc_x_fp_mean, unc_x_tp_mean, unc_y_fp_mean, unc_y_tp_mean, unc_a_fp_mean, unc_a_tp_mean, reg_fp_mean, reg_tp_mean, unc_ale_cls_fp_mean, unc_ale_cls_tp_mean, unc_epi_cls_fp_mean, unc_epi_cls_tp_mean, unc_epi_reg_fp_mean, unc_epi_reg_tp_mean))
            logger.info('The std epistemic cls Uncertainty is %.4f' % unc_epi_cls_tp_std)
            logger.info('The Average tp score is %.4f' % score_tp_mean)
            logger.info('The std tp score is %.4f' % score_tp_std)
            logger.info('The std tp ale cls Uncertainty is %.4f' % unc_ale_cls_tp_std)
            logger.info('The mean all ale reg Uncertainty is %.4f' % ale_reg_all_mean)
            logger.info('The std all ale reg Uncertainty is %.4f' % ale_reg_all_std)
            logger.info('The mean all epi reg Uncertainty is %.4f' % unc_epi_reg_all_mean)
            logger.info('The std all epi reg Uncertainty is %.4f' % unc_epi_reg_all_std)

            logger.info('The mean distance tp is %.4f' % distance_tp)
            logger.info('The mean distance abnormal tp is %.4f' % distance_tp_ab)
            logger.info('The mean distance all is %.4f' % distance_all_mean)
            logger.info('The std distance all is %.4f' % distance_all_std)
            file_handler.flush()
            file_handler.close()
        # 确认日志文件内容
        # with open(unc_log_file, 'r') as f:
        #     print(f.read())


if __name__ == '__main__':
    main()
