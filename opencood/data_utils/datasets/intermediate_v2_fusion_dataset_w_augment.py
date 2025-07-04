# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
import json
import os
import os.path as osp
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def build_idx_to_info(data):
    idx2info = {}
    for elem in data:
        if elem["pointcloud_path"] == "":
            continue
        idx = elem["pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info

def build_idx_to_co_info(data):
    idx2info = {}
    for elem in data:
        if elem["vehicle_pointcloud_path"] == "":
            continue
        idx = elem["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info

def build_inf_fid_to_veh_fid(data):
    inf_fid2veh_fid = {}
    for elem in data:
        veh_fid = elem["vehicle_pointcloud_path"].split("/")[-1].rstrip('.pcd')
        inf_fid = elem["infrastructure_pointcloud_path"].split("/")[-1].rstrip('.pcd')
        inf_fid2veh_fid[inf_fid] = veh_fid
    return inf_fid2veh_fid

def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result

def getIntermediatev2augFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateV2AugFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train) 
            # intermediate and supervise single
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']

            # # self.anchor_box = self.post_processor.generate_anchor_box()
            # # self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.kd_flag = params.get('kd_flag', False)

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']
            
            # # 从父类迁移过来的
            # self.params = params
            # self.visualize = visualize
            # self.train = train

            # self.pre_processor = build_preprocessor(params["preprocess"], train)
            # self.post_processor = build_postprocessor(params["postprocess"], train)
            # self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
            # class_names = params.get('class_names', ['Car'])
            # self.data_augmentor = DataAugmentor(params['data_augment'], train, params['data_dir'], class_names)

            # if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            #     self.clip_pc = True
            # else:
            #     self.clip_pc = False

            # if 'train_params' not in params or 'max_cav' not in params['train_params']:
            #     self.max_cav = 2
            # else:
            #     self.max_cav = params['train_params']['max_cav']

            # self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
            # self.load_camera_file = True if 'camera' in params['input_source'] else False
            # self.load_depth_file = True if 'depth' in params['input_source'] else False

            # assert self.load_depth_file is False

            # self.label_type = params['label_type'] # 'lidar' or 'camera'
            # self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
            #                                             else self.generate_object_center_camera

            # if self.load_camera_file:
            #     self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

            # if self.train:
            #     split_dir = params['root_dir']
            # else:
            #     split_dir = params['validate_dir']

            # self.root_dir = params['data_dir']

            # self.inf_idx2info = build_idx_to_info( # 读取路端标签 形成路端id对应其信息字典的形式
            #     load_json(osp.join(self.root_dir, "infrastructure-side/data_info.json"))
            # )
            # self.co_idx2info = build_idx_to_co_info( # 读取协同标签，形成车端id对应该项协同场景的所有信息的形式
            #     load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
            # ) # 依旧读取协同标签，形成路端id对应车端id的形式，也就是形成了一一对应
            # self.inf_fid2veh_fid = build_inf_fid_to_veh_fid(load_json(osp.join(self.root_dir, "cooperative/data_info.json"))
            # )

            # self.split_info = read_json(split_dir)
            # self.data = []
            # cnt = 0
            # for veh_idx in self.split_info:
            #     if self.is_valid_id(veh_idx):
            #         self.data.append(veh_idx)
            #     else:
            #         cnt += 1
            # if len(self.split_info) == len(self.data):
            #     print("===协同信息无缺失,共 %d 帧==="%len(self.data))
            # if cnt > 0:
            #     print("===协同信息缺失%d帧==="%cnt)
            # self.split_info = self.data
            # self.co_data = self.co_idx2info
            # self.db_num = 0
            self.fade = True # 在最后几个epoch会开启用于调整对于test中的数据分布的适应  TODO 为True的时候发现会报错，会导致问题协同训练warp feature报错
            if self.train is False:
                self.fade = True # 如果不在训练状态，即在验证以及测试阶段，那会关闭data augment 此时fade必须为True从而避免获取一些根本无法获取的data augment数据
            if self.fade:
                print("===Fade strategy===")

            # co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
            # self.co_data = OrderedDict()
            # for frame_info in co_datainfo:
            #     veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            #     self.co_data[veh_frame_id] = frame_info
                
            # if "noise_setting" not in self.params:
            #     self.params['noise_setting'] = OrderedDict()
            #     self.params['noise_setting']['add_noise'] = False
                
            if "dataset" in self.params:
                self.dataset_name = self.params['dataset']
                print(f"=== dataset name is {self.dataset_name} ===")
            else:
                raise ValueError("we must provide the name of dataset!")


        def is_valid_id(self, veh_frame_id):
            frame_info = {}
            
            frame_info = self.co_idx2info[veh_frame_id] # 取出协同信息
            inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "") # 当前路端的帧id
            cur_inf_info = self.inf_idx2info[inf_frame_id] # 取出路端信息
            delay_id = id_to_str(int(inf_frame_id)) 
            if delay_id not in self.inf_fid2veh_fid.keys(): # 必须有车路对应
                return False

            return True

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)

            '''
            加了 gt sampling后不太好直接监督single, 所以我们直接使用协同label作为gt
            XXX 反思一下这个时候到底要用协同标签还是single 标签?
            用协同label 则更好说明了lidar范围内的检测情况
            用单车label 则更好表现了lidar范围内有效标注的检测情况
            '''
            
            # note the reference pose ego 这里是处理only ego的情况，但是仍然要使用协同标签 XXX 这是是否要使用协同标签？
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                        ego_pose_clean)

            # # generate targets label single GT, note the reference pose is itself. XXX 不太理解，按理说是用来监督单车的，但是generate_object_center其实用的协同标签
            # object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single( # FIXME 对比了一下where2comm的源码，这里似乎确实有问题, 测试where2comm这种需要单车监督的模型要小心
            #     [selected_cav_base], selected_cav_base['params']['lidar_pose']
            # )

            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                # data augmentation for single
                if selected_cav_base['ego']:
                    if self.fade is False:
                        lidar_np, object_bbx_center, object_bbx_mask, sampled_boxes = self.augment(lidar_np, object_bbx_center, object_bbx_mask, return_sampled_boxes=True,
                                                                                    flip=selected_cav_base['flip'],
                                                                                    rotation=selected_cav_base['noise_rotation'],
                                                                                    scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling
                    else:
                        lidar_np = lidar_np
                                                                                
                    projected_lidar = copy.deepcopy(lidar_np)[:,:3] # 只有单车 不必空间变换
                else:
                    print('fuck!')
                    raise ValueError("fuck!")

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar

                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})



            # camera
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    

                selected_cav_processed.update(
                    {
                    "image_inputs": 
                        {
                            "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            # # note the reference pose ego 这里是处理only ego的情况，但是仍然要使用协同标签 XXX 这是是否要使用协同标签？
            # object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
            #                                             ego_pose_clean)

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )

            # 协同的性能用协同标签去监督，而单车标签仍用单车标签，注意将协同标签中加入的sampled gt给补到单车标签中，因为点云已经发生变化了
            object_bbx_center, object_bbx_mask, _ = self.generate_object_center_single( # FIXME 对比了一下where2comm的源码，这里似乎确实有问题, 测试where2comm这种需要单车监督的模型要小心
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            if self.fade is False:
                # 做数据增强的原因是存在翻转、旋转以及缩放，这些是在协同标签以及点云上做了的，所以要把单车标签也操作一下，这只针对single 标签，因此点云随意传入即可，返回也用不到
                _, object_bbx_center, object_bbx_mask = self.augment(lidar_np, object_bbx_center, object_bbx_mask, choice=1, # 单车不需要GT sampling，直接将协同中sample的加入就行，所以跳过
                                                                                            flip=selected_cav_base['flip'],
                                                                                            rotation=selected_cav_base['noise_rotation'],
                                                                                            scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling
                # 车端single只需要将多出来的gt拼起来即可
                gt_boxes_valid = object_bbx_center[object_bbx_mask == 1]
                gt_boxes_valid = np.concatenate([gt_boxes_valid, sampled_boxes], axis=0)
                object_bbx_center[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
                object_bbx_mask[:gt_boxes_valid.shape[0]] = 1

            selected_cav_processed.update({
                                "single_object_bbx_center": object_bbx_center,
                                "single_object_bbx_mask": object_bbx_mask})

            return selected_cav_processed

        def get_all_gts(self, selected_cav_base, ego_cav_base):
            """
            just for datasets none dairv2x
            """
            selected_cav_processed = {}

            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            cav_bbx_center, cav_bbx_mask, cav_ids = self.generate_object_center([selected_cav_base], ego_pose_clean)
 
            selected_cav_processed.update(
                {
                    "object_bbx_center": cav_bbx_center[cav_bbx_mask == 1],
                    "object_bbx_mask": cav_bbx_mask,
                    "object_ids": cav_ids,
                }
            )

            return selected_cav_processed

        def get_item_single_common(self, selected_cav_base, ego_cav_base, cooperative_bbx_center, cooperative_bbx_mask, stay_static):
            """
            Process a single CAV's information for the train/test pipeline.
            designed for v2xset/opv2v/v2v4real/v2xreal

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation
            stay_static: whether to keep the same config of gt sampling
            cooperative: control get cooperative info or single info

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}

            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            cav_bbx_center, cav_bbx_mask, cav_ids = cooperative_bbx_center, cooperative_bbx_mask, None
            # cav_bbx_center, cav_bbx_mask, cav_ids = self.generate_object_center([selected_cav_base], ego_pose_clean)
            # # TODO 以下是针对DairV2X特异化处理的，因为协同标签只加载在车端，但是路端也需要处理，注意，这个后面要回退，因为在推理时生成GT会用两者标签并集去重，协同标签确实只用加载在车端
            # selected_inf_base['params']['vehicles_all'] = selected_cav_base['params']['vehicles_all'] # 这里必须要把协同标签拿过来，因为在Dair中只有车端有协同标签

            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np_cav = selected_cav_base['lidar_np']
                lidar_np_cav = shuffle_points(lidar_np_cav)
                lidar_np_cav = mask_ego_points(lidar_np_cav)# remove points that hit itself

                # data augmentation 
                random_seed = np.random.randint(100000) #for same random augmentation
                # gt sampling 中使用的fusion sample, 是基于ego 坐标系生成的 所以先投影过去
                projected_lidar_cav = box_utils.project_points_by_matrix_torch(lidar_np_cav[:, :3], transformation_matrix)# project the lidar to ego space
                lidar_proj_np_cav = copy.deepcopy(lidar_np_cav)
                lidar_proj_np_cav[:,:3] = projected_lidar_cav # 点云变为投影后的点云 这里使用的是协同标签
                if self.fade is False:
                    lidar_np_cav, cav_bbx_center, cav_bbx_mask, sampled_boxes_cav = self.augment(lidar_proj_np_cav, cooperative_bbx_center, cooperative_bbx_mask, random_seed, stay_static=stay_static, return_sampled_boxes=True,
                                                                                                flip=selected_cav_base['flip'],
                                                                                                rotation=selected_cav_base['noise_rotation'],
                                                                                                scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling
                    # print("======================================================")
                    # print("sampled_boxes_cav shape is ", sampled_boxes_cav.shape)
                    # print("cooperative_bbx_center[cooperative_bbx_mask==1] shape is ", cooperative_bbx_center[cooperative_bbx_mask==1].shape)
                    # print("======================================================")

                else:
                    lidar_np_cav = lidar_proj_np_cav

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar_cav})

                if self.kd_flag:
                    selected_cav_processed.update({'projected_lidar': lidar_np_cav}) #lidar_proj_np_cav

                # 原来为了采样gt 需要投影到ego 现在需要还原后再进行体素化
                transformation_matrix_inv = x1_to_x2(ego_pose, selected_cav_base['params']['lidar_pose']) 
                projected_back_lidar_inf = box_utils.project_points_by_matrix_torch(lidar_np_cav[:, :3], transformation_matrix_inv)# project the lidar to ego space
                lidar_proj_np_cav = copy.deepcopy(lidar_np_cav)
                lidar_proj_np_cav[:, :3] = projected_back_lidar_inf
                processed_lidar_cav = self.pre_processor.preprocess(lidar_proj_np_cav) # 体素化 重新投影回去

                selected_cav_processed.update({'processed_features': processed_lidar_cav})

            selected_cav_processed.update(
                {
                    "object_bbx_center": cav_bbx_center[cav_bbx_mask == 1], # 按理说每一个cav调用进来这里都是处理的协同标签
                    "object_bbx_mask": cav_bbx_mask,
                    "object_ids": cav_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean,
                }
            )

            '''
            interesting! 所有的agent的点云都投影到ego坐标系中, 然后进行gt sampling和augment 随后投影回去作为作为感知输入
            在体素预处理中, 会根据lidar的范围剔除特征图范围外的点, 然而对于单车监督 还需要将其单车标签也更新
            '''
            # generate targets label single GT, note the reference pose is itself.
            cav_bbx_center, cav_bbx_mask, _ = self.generate_object_center_single(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            if self.fade is False:
                # 做数据增强的原因是存在翻转、旋转以及缩放，这些是在协同标签以及点云上做了的，所以要把单车标签也操作一下，这只针对single 标签，因此点云随意传入即可，返回也用不到
                _, cav_bbx_center, cav_bbx_mask = self.augment(lidar_np_cav, cav_bbx_center, cav_bbx_mask, random_seed, choice=1,
                                                                                            flip=selected_cav_base['flip'],
                                                                                            rotation=selected_cav_base['noise_rotation'],
                                                                                            scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling

                # 车端single 需要将采样的gt boxes 投影回single view 然后去除范围外的部分
                gt_boxes_valid = cav_bbx_center[cav_bbx_mask == 1]
                if sampled_boxes_cav.shape[0] > 0:
                    sampled_boxes_corner_cav = box_utils.boxes_to_corners_3d(sampled_boxes_cav, 'lwh')
                    projected_sampled_boxes_corner_cav = box_utils.project_box3d(sampled_boxes_corner_cav, transformation_matrix_inv)
                    projected_sampled_boxes_corner_cav = box_utils.mask_boxes_outside_range_numpy(projected_sampled_boxes_corner_cav, self.params['postprocess']['gt_range'], order='lwh')
                    sampled_boxes_cav = box_utils.corners_to_boxes_3d(projected_sampled_boxes_corner_cav, 'lwh') # (extra_num, 7)
                gt_boxes_valid = np.concatenate([gt_boxes_valid, sampled_boxes_cav], axis=0)
                cav_bbx_center[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
                cav_bbx_mask[:gt_boxes_valid.shape[0]] = 1

            selected_cav_processed.update({
                                "single_object_bbx_center": cav_bbx_center,
                                "single_object_bbx_mask": cav_bbx_mask})

            # camera
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    

                selected_cav_processed.update(
                    {
                    "image_inputs": 
                        {
                            "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            return selected_cav_processed

        def get_item_all_agent(self, selected_cav_base, selected_inf_base, ego_cav_base):
            """
            Process all CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
            selected_inf_base : dict
                The dictionary contains a single CAV's raw information.
            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            selected_inf_processed : dict
                The dictionary contains the inf's processed information.
            """
            selected_cav_processed = {}
            selected_inf_processed = {}

            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix for cav
            transformation_matrix_cav = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose)
            transformation_matrix_clean_cav = x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)
            cav_bbx_center, cav_bbx_mask, cav_ids = self.generate_object_center([selected_cav_base], ego_pose_clean)
            # calculate the transformation matrix for inf
            transformation_matrix_inf = x1_to_x2(selected_inf_base['params']['lidar_pose'], ego_pose) 
            transformation_matrix_clean_inf = x1_to_x2(selected_inf_base['params']['lidar_pose_clean'], ego_pose_clean)
            # TODO 以下是针对DairV2X特异化处理的，因为协同标签只加载在车端，但是路端也需要处理，注意，这个后面要回退，因为在推理时生成GT会用两者标签并集去重，协同标签确实只用加载在车端
            selected_inf_base['params']['vehicles_all'] = selected_cav_base['params']['vehicles_all'] # 这里必须要把协同标签拿过来，因为在Dair中只有车端有协同标签
            inf_bbx_center, inf_bbx_mask, inf_ids = self.generate_object_center([selected_inf_base], ego_pose_clean)

            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar for cav
                lidar_np_cav = selected_cav_base['lidar_np']
                lidar_np_cav = shuffle_points(lidar_np_cav)
                lidar_np_cav = mask_ego_points(lidar_np_cav)# remove points that hit itself
                # process lidar for inf
                lidar_np_inf = selected_inf_base['lidar_np']
                lidar_np_inf = shuffle_points(lidar_np_inf)
                lidar_np_inf = mask_ego_points(lidar_np_inf)
                # data augmentation for both cav and inf 
                random_seed = np.random.randint(100000) #for same random augmentation
                # gt sampling 中使用的fusion sample, 是基于ego 坐标系生成的
                projected_lidar_cav = box_utils.project_points_by_matrix_torch(lidar_np_cav[:, :3], transformation_matrix_cav)# project the lidar to ego space
                lidar_proj_np_cav = copy.deepcopy(lidar_np_cav)
                lidar_proj_np_cav[:,:3] = projected_lidar_cav # 点云变为投影后的点云 这里使用的是协同标签
                if self.fade is False:
                    lidar_np_cav, cav_bbx_center, cav_bbx_mask, sampled_boxes_cav = self.augment(lidar_proj_np_cav, cav_bbx_center, cav_bbx_mask, random_seed, stay_static=True, return_sampled_boxes=True,
                                                                                                flip=selected_cav_base['flip'],
                                                                                                rotation=selected_cav_base['noise_rotation'],
                                                                                                scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling
                else:
                    lidar_np_cav = lidar_proj_np_cav
                # print("cav_bbx_center[cav_bbx_mask == 1] shape is ", cav_bbx_center[cav_bbx_mask == 1].shape)

                projected_lidar_inf = box_utils.project_points_by_matrix_torch(lidar_np_inf[:, :3], transformation_matrix_inf)# project the lidar to ego space
                lidar_proj_np_inf = copy.deepcopy(lidar_np_inf)
                lidar_proj_np_inf[:,:3] = projected_lidar_inf
                # print("before aug lidar_proj_np_inf shape is ", lidar_proj_np_inf.shape)
                # print("before aug inf_bbx_center[inf_bbx_mask == 1] shape is ", inf_bbx_center[inf_bbx_mask == 1].shape)
                # 路端不需要增强，本质上是ego在做协同 所以只增强ego的gt  协同标签被扩增，那路端single标签怎么办? TODO single 路端已经加入了采样的实例点云，那还需要把增加的box取回增加到单车
                if self.fade is False:
                    lidar_np_inf, inf_bbx_center, inf_bbx_mask, sampled_boxes_inf = self.augment(lidar_proj_np_inf, inf_bbx_center, inf_bbx_mask, random_seed, return_sampled_boxes=True,
                                                                                                flip=selected_cav_base['flip'],
                                                                                                rotation=selected_cav_base['noise_rotation'],
                                                                                                scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling                    
                else:
                    lidar_np_inf = lidar_proj_np_inf
                # print("after aug inf_bbx_center[inf_bbx_mask == 1] shape is ", inf_bbx_center[inf_bbx_mask == 1].shape)
                # print("after aug lidar_proj_np_inf shape is ", lidar_proj_np_inf.shape)

                # if self.db_num > 10:
                #     exit()
                # else:
                #     self.db_num += 1

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar_cav})
                    selected_inf_processed.update({'projected_lidar': projected_lidar_inf})

                if self.kd_flag:
                    selected_cav_processed.update({'projected_lidar': lidar_np_cav}) #lidar_proj_np_cav
                    selected_inf_processed.update({'projected_lidar': lidar_np_inf})

                transformation_matrix_inf_inv = x1_to_x2(ego_pose, selected_inf_base['params']['lidar_pose']) 
                projected_back_lidar_inf = box_utils.project_points_by_matrix_torch(lidar_np_inf[:, :3], transformation_matrix_inf_inv)# project the lidar to ego space
                lidar_proj_np_inf = copy.deepcopy(lidar_np_inf)
                lidar_proj_np_inf[:,:3] = projected_back_lidar_inf
                processed_lidar_cav = self.pre_processor.preprocess(lidar_np_cav) # 体素化
                processed_lidar_inf = self.pre_processor.preprocess(lidar_proj_np_inf) # 重新投影回去

                selected_cav_processed.update({'processed_features': processed_lidar_cav})
                selected_inf_processed.update({'processed_features': processed_lidar_inf})

            selected_inf_base['params']['vehicles_all'] = [] # 路端协同标签

            selected_cav_processed.update(
                {
                    "object_bbx_center": cav_bbx_center[cav_bbx_mask == 1],
                    "object_bbx_mask": cav_bbx_mask,
                    "object_ids": cav_ids,
                    'transformation_matrix': transformation_matrix_cav,
                    'transformation_matrix_clean': transformation_matrix_clean_cav,
                }
            )
            selected_inf_processed.update(
                {
                    "object_bbx_center": inf_bbx_center[inf_bbx_mask == 1],
                    "object_bbx_mask": inf_bbx_mask,
                    "object_ids": inf_ids,
                    'transformation_matrix': transformation_matrix_inf,
                    'transformation_matrix_clean': transformation_matrix_clean_inf,
                }
            )

            '''
            interesting! ego 我们做gt sampling, 但是other agent 我们没做, 所以ego的单车监督用协同标签
            而other agents 我们使用其自身的single 标签
            '''
            # generate targets label single GT, note the reference pose is itself. XXX 不太理解，按理说是用来监督单车的，但是generate_object_center其实用的协同标签
            cav_bbx_center, cav_bbx_mask, _ = self.generate_object_center_single( # FIXME 对比了一下where2comm的源码，这里似乎确实有问题, 测试where2comm这种需要单车监督的模型要小心
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            inf_bbx_center, inf_bbx_mask, _ = self.generate_object_center_single( # NOTE infra side 也要进行单车监督，需要其single label
                [selected_inf_base], selected_inf_base['params']['lidar_pose']
            )
            if self.fade is False:
                # 做数据增强的原因是存在翻转、旋转以及缩放，这些是在协同标签以及点云上做了的，所以要把单车标签也操作一下，这只针对single 标签，因此点云随意传入即可，返回也用不到
                _, cav_bbx_center, cav_bbx_mask = self.augment(lidar_np_cav, cav_bbx_center, cav_bbx_mask, random_seed, choice=1,
                                                                                            flip=selected_cav_base['flip'],
                                                                                            rotation=selected_cav_base['noise_rotation'],
                                                                                            scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling
                _, inf_bbx_center, inf_bbx_mask = self.augment(lidar_proj_np_inf, inf_bbx_center, inf_bbx_mask, random_seed, choice=1,
                                                                                            flip=selected_cav_base['flip'],
                                                                                            rotation=selected_cav_base['noise_rotation'],
                                                                                            scale=selected_cav_base['noise_scale']) #choice=1 skip gt_sampling
                # 车端single只需要将多出来的gt拼起来即可
                gt_boxes_valid = cav_bbx_center[cav_bbx_mask == 1]
                gt_boxes_valid = np.concatenate([gt_boxes_valid, sampled_boxes_cav], axis=0)
                cav_bbx_center[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
                cav_bbx_mask[:gt_boxes_valid.shape[0]] = 1

                # 路端single首先要变换回去，然后判断范围，最后仍旧保留才会拼接到路端single的标签
                gt_boxes_valid = inf_bbx_center[inf_bbx_mask == 1]
                if sampled_boxes_inf.shape[0] > 0:
                    sampled_boxes_corner_inf = box_utils.boxes_to_corners_3d(sampled_boxes_inf, 'lwh')
                    projected_sampled_boxes_corner_inf = box_utils.project_box3d(sampled_boxes_corner_inf, transformation_matrix_inf_inv)
                    projected_sampled_boxes_corner_inf = box_utils.mask_boxes_outside_range_numpy(projected_sampled_boxes_corner_inf, self.params['postprocess']['gt_range'], order='lwh')
                    # projected_sampled_boxes_corner_inf = projected_sampled_boxes_corner_inf[mask, :, :]
                    sampled_boxes_inf = box_utils.corners_to_boxes_3d(projected_sampled_boxes_corner_inf, 'lwh') # (extra_num, 7)
                gt_boxes_valid = np.concatenate([gt_boxes_valid, sampled_boxes_inf], axis=0)
                inf_bbx_center[:gt_boxes_valid.shape[0], :] = gt_boxes_valid
                inf_bbx_mask[:gt_boxes_valid.shape[0]] = 1

            # _, inf_bbx_center, inf_bbx_mask = self.augment(lidar_proj_np_inf, inf_bbx_center, inf_bbx_mask, random_seed, choice=1)

            selected_cav_processed.update({
                                "single_object_bbx_center": cav_bbx_center,
                                "single_object_bbx_mask": cav_bbx_mask})

            selected_inf_processed.update({
                                "single_object_bbx_center": inf_bbx_center,
                                "single_object_bbx_mask": inf_bbx_mask})
            # camera
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                    

                selected_cav_processed.update(
                    {
                    "image_inputs": 
                        {
                            "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            return selected_cav_processed, selected_inf_processed

        # modify from V2V4Real by xuyunjiang 2025-01-11
        def generate_augment(self):
            flip = [None, None, None]
            noise_rotation = None
            noise_scale = None

            for aug_ele in self.params['data_augment']:
                # for intermediate fusion only
                if 'random_world_rotation' in aug_ele['NAME']:
                    rot_range = \
                        aug_ele['WORLD_ROT_ANGLE']
                    if not isinstance(rot_range, list):
                        rot_range = [-rot_range, rot_range]
                    noise_rotation = np.random.uniform(rot_range[0],
                                                            rot_range[1]) # 均匀采样一个旋转角度

                if 'random_world_flip' in aug_ele['NAME']:
                    for i, cur_axis in enumerate(aug_ele['ALONG_AXIS_LIST']):
                        enable = np.random.choice([False, True], replace=False,
                                                p=[0.5, 0.5])
                        flip[i] = enable # 是否翻转

                if 'random_world_scaling' in aug_ele['NAME']:
                    scale_range = \
                        aug_ele['WORLD_SCALE_RANGE']
                    noise_scale = \
                        np.random.uniform(scale_range[0], scale_range[1]) # 均匀采样一个缩放比例

            return flip, noise_rotation, noise_scale

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            # augmentation related
            flip, noise_rotation, noise_scale = self.generate_augment()

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    camera0_file = cav_content['vis']['camera0']
                    infra_pic_file = cav_content['vis']['infra_pic']
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agents_image_inputs = []
            processed_features = []
            coop_object_stack = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = [] # disconet

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    # print("距离太远,距离为: %0.4f"%distance)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            for cav_id in too_far:
                base_data_dict.pop(cav_id)

            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list'] # include those out of range
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list] # indexing current agent in `all_agent_id_list`

                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                        cur_agnet_pose, 
                                                                        uncertainty_list=uncertainty_list, 
                                                                        **self.box_align_args)
                        cur_agnet_pose[:,[0,1,4]] = refined_pose 

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()



            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list) # 1 or 2 Dair中就这几种可能
            
            if self.dataset_name == 'dairv2x':
                if cav_num == 1:
                    selected_cav_base = base_data_dict[0] # ego的 信息
                    selected_cav_base['flip'] = flip
                    selected_cav_base['noise_rotation'] = noise_rotation
                    selected_cav_base['noise_scale'] = noise_scale
                    selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)
                        
                    object_stack.append(selected_cav_processed['object_bbx_center'])
                    object_id_stack += selected_cav_processed['object_ids']
                    if self.load_lidar_file:
                        processed_features.append(selected_cav_processed['processed_features'])

                    if self.visualize or self.kd_flag:
                        projected_lidar_stack.append(selected_cav_processed['projected_lidar'])

                    if self.supervise_single: # 用于监督单车检测结果用的，比如说where2comm就这样做
                        single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                        single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
                else:
                    base_data_dict[0]['flip'] = flip
                    base_data_dict[0]['noise_rotation'] = noise_rotation
                    base_data_dict[0]['noise_scale'] = noise_scale
                    selected_cav_processed, selected_inf_processed = self.get_item_all_agent(base_data_dict[0], base_data_dict[1], ego_cav_base)           
                    object_stack.append(selected_cav_processed['object_bbx_center'])
                    object_stack.append(selected_inf_processed['object_bbx_center'])
                    object_id_stack += selected_cav_processed['object_ids']
                    object_id_stack += selected_inf_processed['object_ids']
                    if self.load_lidar_file:
                        processed_features.append(selected_cav_processed['processed_features'])
                        processed_features.append(selected_inf_processed['processed_features'])

                    if self.visualize or self.kd_flag:
                        projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                        projected_lidar_stack.append(selected_inf_processed['projected_lidar'])

                    if self.supervise_single: # 用于监督单车检测结果用的，比如说where2comm就这样做
                        # single_label_list.append(selected_cav_processed['single_label_dict'])
                        single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                        single_object_bbx_center_list.append(selected_inf_processed['single_object_bbx_center'])
                        single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
                        single_object_bbx_mask_list.append(selected_inf_processed['single_object_bbx_mask'])

                # generate single view GT label
                if self.supervise_single:
                    single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list)) # （N_cav, max_num, 7）
                    single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                    processed_data_dict['ego'].update({
                        # "single_label_dict_torch": single_label_dicts,
                        "single_object_bbx_center_torch": single_object_bbx_center,
                        "single_object_bbx_mask_torch": single_object_bbx_mask,
                        })

                if self.kd_flag:
                    stack_lidar_np = np.vstack(projected_lidar_stack)
                    stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                                self.params['preprocess'][
                                                    'cav_lidar_range'])
                    stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                    processed_data_dict['ego'].update({'teacher_processed_lidar':
                    stack_feature_processed})
            else:
                # 先把所有标签取出，合并成协同标签
                for i, cav_id in enumerate(cav_id_list):
                    selected_cav_base = base_data_dict[cav_id]
                    selected_cav_processed = self.get_all_gts(selected_cav_base, ego_cav_base)
                    object_stack.append(selected_cav_processed['object_bbx_center'])
                    object_id_stack += selected_cav_processed['object_ids']
            
            # exclude all repetitive objects    
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]
            # print("object_id_stack is ", object_id_stack)
            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            # print("数据增强前的协同标签: ", object_stack)
            # 不同于dairv2x数据集将协同标签记录在车端，其他很多数据集都是将自车标签去重合并后得到的标签
            if self.dataset_name != "dairv2x":
                # print("len(cav_id_list) is ", len(cav_id_list))
                for cnt, cav_id in enumerate(cav_id_list):
                    gt_sampling_keep_same = True
                    if cnt == len(cav_id_list)-1: # 这是让同一个场景下的gt采样一致，最后一个agent要负责将gt采样设置更新，从而让下一个场景可以重新采样
                        gt_sampling_keep_same = False
                    selected_cav_base = base_data_dict[cav_id]
                    selected_cav_base['flip'] = flip
                    selected_cav_base['noise_rotation'] = noise_rotation
                    selected_cav_base['noise_scale'] = noise_scale
                    cooperative_bbx_center = copy.deepcopy(object_bbx_center)
                    cooperative_bbx_mask = copy.deepcopy(mask)
                    selected_cav_processed = self.get_item_single_common(selected_cav_base, ego_cav_base, 
                                                                         cooperative_bbx_center=cooperative_bbx_center, 
                                                                         cooperative_bbx_mask=cooperative_bbx_mask, 
                                                                         stay_static=gt_sampling_keep_same)
                    # print("selected_cav_processed['object_bbx_center']  is ", selected_cav_processed['object_bbx_center'])
                    # object_stack.append(selected_cav_processed['object_bbx_center'])
                    # object_id_stack += selected_cav_processed['object_ids']
                    if self.load_lidar_file:
                        # print("selected_cav_processed['processed_features']['voxel_features'] shape is ", selected_cav_processed['processed_features']['voxel_features'].shape)
                        processed_features.append(
                            selected_cav_processed['processed_features'])
                    if self.load_camera_file:
                        agents_image_inputs.append(
                            selected_cav_processed['image_inputs'])
                    if self.visualize or self.kd_flag:
                        projected_lidar_stack.append(
                            selected_cav_processed['projected_lidar'])
                    if self.supervise_single:
                        single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                        single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

                # 协同标签会在处理后多出一些采样的gt标签
                coop_object_stack.append(selected_cav_processed['object_bbx_center']) # 这里默认就是用的循环最后一个的结果，但是其实由于我传入的是统一的协同标签，所有这两个只需要一次就行，其余的每一次都是一样
                coop_object_stack = np.vstack(coop_object_stack)
                object_bbx_center = \
                    np.zeros((self.params['postprocess']['max_num'], 7))
                mask = np.zeros(self.params['postprocess']['max_num'])
                object_bbx_center[:coop_object_stack.shape[0], :] = coop_object_stack
                mask[:coop_object_stack.shape[0]] = 1
                # print("数据增强后的协同标签: ", coop_object_stack)
                # xxx
                # generate single view GT label
                if self.supervise_single:
                    single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                    single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                    processed_data_dict['ego'].update({
                        "single_object_bbx_center_torch": single_object_bbx_center,
                        "single_object_bbx_mask_torch": single_object_bbx_mask,
                        })
                    
                if self.kd_flag:
                    stack_lidar_np = np.vstack(projected_lidar_stack)
                    stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                                self.params['preprocess'][
                                                    'cav_lidar_range'])
                    stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                    processed_data_dict['ego'].update({'teacher_processed_lidar':
                    stack_feature_processed})

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})
            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})


            # generate targets label
            # label_dict = \
            #     self.post_processor.generate_label(
            #         gt_box_center=object_bbx_center,
            #         anchors=self.anchor_box,
            #         mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                # 'anchor_box': self.anchor_box,
                # 'label_dict': label_dict,
                'cav_num': cav_num,
                'camera0_file': camera0_file, # 可视化前摄用的
                'infra_pic_file': infra_pic_file,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})

            # print("object_bbx_center[object_bbx_mask == 1].shape is ", object_bbx_center[mask == 1].shape)
            # print("processed_data_dict['ego']['object_ids'] is ", processed_data_dict['ego']['object_ids'])
            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})
                split_num = []
                for pd in projected_lidar_stack:
                    split_num.append(pd.shape[0])
                processed_data_dict['ego'].update({'origin_lidar_splitnum':
                    split_num}) # [n1, n2, ...]

            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})

            return processed_data_dict


        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            origin_lidar_splitnum = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                # pos_equal_one_single = []
                # neg_equal_one_single = []
                # targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                record_len.append(ego_dict['cav_num'])
                # label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])
                    origin_lidar_splitnum.append(ego_dict['origin_lidar_splitnum'])

                if self.kd_flag:
                    teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    # pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    # neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    # targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])


            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))

            # for centerpoint
            # label_torch_dict.update({'object_bbx_center': object_bbx_center,
            #                          'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            # label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            # label_torch_dict['record_len'] = record_len
            

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    # 'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_pose': lidar_pose,
                                    # 'anchor_box': self.anchor_box_torch
                                    })


            if self.visualize:
                # origin_lidar = \
                #     np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = \
                    np.array(origin_lidar) # 不要打乱，否则将无法分别可视化每个agent的点云 2024年8月16日 xyj
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})
                output_dict['ego'].update({'origin_lidar_splitnum': origin_lidar_splitnum})


            if self.kd_flag:
                teacher_processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(teacher_processed_lidar_list)
                output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})


            if self.supervise_single:
                output_dict['ego'].update({
                    # "label_dict_single":{
                    #         "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                    #         "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                    #         "targets": torch.cat(targets_single, dim=0),
                    #         # for centerpoint
                    #         "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    #         "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                    #     },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })


            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            # if batch[0]['ego']['anchor_box'] is not None:
            #     output_dict['ego'].update({'anchor_box':
            #         self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                        'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            output_dict['ego'].update({
                'camera0_file': batch[0]['ego']['camera0_file'], # 可视化前摄用的
                'infra_pic_file': batch[0]['ego']['infra_pic_file'],
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list']
            })

            return output_dict


        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            # pred_box_tensor, pred_score = \
            #     self.post_processor.post_process(data_dict, output_dict)
            pred_box_tensor = output_dict["ego"]['pred_boxes']
            pred_score = output_dict["ego"]['pred_scores']
            from opencood.utils import box_utils
            pred_box_tensor = box_utils.boxes_to_corners_3d(pred_box_tensor, order='lwh')

            # keep_index_1 = box_utils.remove_large_pred_bbx(pred_box_tensor)
            keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box_tensor, zmin=-3.5, zmax=1.5)
            keep_index_1 = keep_index_2 # 较大的bbx 先不移除
            keep_index = torch.logical_and(keep_index_1, keep_index_2)

            pred_box_tensor = pred_box_tensor[keep_index]
            pred_score = pred_score[keep_index]

            keep_index = box_utils.nms_rotated(pred_box_tensor,
                                            pred_score,
                                            0.15
                                            )

            pred_box_tensor = pred_box_tensor[keep_index]

            # select cooresponding score
            pred_score = pred_score[keep_index]
            
            # filter out the prediction out of the range. with z-dim
            pred_box3d_np = pred_box_tensor.cpu().numpy()
            pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(pred_box3d_np,
                                                        self.params['preprocess']['cav_lidar_range'],
                                                        order='lwh',
                                                        return_mask=True)
            pred_box_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box_tensor.device)
            pred_score = pred_score[mask]

            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor


    return IntermediateV2AugFusionDataset


