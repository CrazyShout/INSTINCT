# late fusion dataset
import random
import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)


def getLatev2FusionDataset(cls):
    """
    cls: the Basedataset.
    pepare for query based method -- xuyunjiang 2024-11-17
    """
    class LateV2FusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # self.anchor_box = self.post_processor.generate_anchor_box()
            # self.anchor_box_torch = torch.from_numpy(self.anchor_box)
            self.point_cloud_range = self.params['preprocess']['cav_lidar_range']
            self.code_size = self.params['model']['args']['dense_head']['code_size']
            if "dataset" in self.params:
                self.dataset_name = self.params['dataset']
                print(f"=== dataset name is {self.dataset_name} ===")
            else:
                raise ValueError("we must provide the name of dataset!")
            
        def encode_bbox(self, bboxes): # 输入的是n, 7
            z_normalizer = 10
            targets = np.zeros([bboxes.shape[0], self.code_size], dtype=np.float32) # n, 7 同时这里有一个隐式数据类型转变将target 类型从torch.float64变成torch.float32 如果没有这个过程，会报各种类型错误
            targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (
                    self.point_cloud_range[3] - self.point_cloud_range[0])
            targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (
                    self.point_cloud_range[4] - self.point_cloud_range[1])
            targets[:, 2] = (bboxes[:, 2] + z_normalizer) / (2 * z_normalizer) # -10 到 10之间
            targets[:, 3] = bboxes[:, 3] / (self.point_cloud_range[3] - self.point_cloud_range[0])
            targets[:, 4] = bboxes[:, 4] / (self.point_cloud_range[4] - self.point_cloud_range[1])
            targets[:, 5] = bboxes[:, 5] / (2 * z_normalizer)
            targets[:, 6] = (bboxes[:, 6] + np.pi) / (np.pi * 2)
            if self.code_size > 7:
                targets[:, 7] = (bboxes[:, 7]) / (self.point_cloud_range[3] - self.point_cloud_range[0])
                targets[:, 8] = (bboxes[:, 8]) / (self.point_cloud_range[4] - self.point_cloud_range[1])

            return targets

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            if self.train:
                reformat_data_dict = self.get_item_train(base_data_dict)
            else:
                reformat_data_dict = self.get_item_test(base_data_dict, idx)

            return reformat_data_dict

        def get_item_train(self, base_data_dict):
            processed_data_dict = OrderedDict()
            base_data_dict = add_noise_data_dict(
                base_data_dict, self.params["noise_setting"]
            )
            # during training, we return a random cav's data
            # only one vehicle is in processed_data_dict
            if not self.visualize:
                selected_cav_id, selected_cav_base = random.choice(
                    list(base_data_dict.items())
                )
            else:
                selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]
            
            selected_cav_processed = self.get_item_single_car(selected_cav_base)
            processed_data_dict.update({"ego": selected_cav_processed})

            return processed_data_dict


        def get_item_test(self, base_data_dict, idx):
            """
                processed_data_dict.keys() = ['ego', "650", "659", ...]
            """
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            ego_id = -1
            ego_lidar_pose = []
            cav_id_list = []
            lidar_pose_list = []

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items(): # 对于dair： 0：{} 1:{}
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                    break

            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > self.params['comm_range']:
                    continue
                cav_id_list.append(cav_id)
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])

            cav_id_list_newname = []
            for cav_id in cav_id_list: # 对于dair，最多就是2 分别为0或者1
                selected_cav_base = base_data_dict[cav_id]
                # find the transformation matrix from current cav to ego.
                cav_lidar_pose = selected_cav_base['params']['lidar_pose']
                transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
                cav_lidar_pose_clean = selected_cav_base['params']['lidar_pose_clean']
                transformation_matrix_clean = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

                selected_cav_processed = \
                    self.get_item_single_car(selected_cav_base)
                selected_cav_processed.update({'transformation_matrix': transformation_matrix,
                                            'transformation_matrix_clean': transformation_matrix_clean})
                update_cav = "ego" if cav_id == ego_id else cav_id
                processed_data_dict.update({update_cav: selected_cav_processed})
                cav_id_list_newname.append(update_cav)
            
            processed_data_dict['ego']['idx'] = idx
            processed_data_dict['ego']['cav_list'] = cav_id_list_newname

            return processed_data_dict
            '''
            dair-v2x:
            processed_data_dict{
                'ego':{
                    selected_cav_processed:{
                    'processed_lidar':体素信息
                    'object_bbx_center':
                    'object_bbx_mask':
                    'object_ids':
                    'label_dict': 注意, 这里是single label
                    },
                    'idx':idx样本id
                    'cav_list': ['ego', 1]
                }
                1:{selected_cav_processed,
                }
            }
            '''

        def get_item_single_car(self, selected_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}

            # label
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
                [selected_cav_base], selected_cav_base["params"]["lidar_pose_clean"]
            ) # single label 单车标签

            # lidar
            if self.load_lidar_file or self.visualize:
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                lidar_np = mask_points_by_range(lidar_np,
                                                self.params['preprocess'][
                                                    'cav_lidar_range'])
                # remove points that hit ego vehicle
                lidar_np = mask_ego_points(lidar_np)

                # data augmentation, seems very important for single agent training, because lack of data diversity.
                # only work for lidar modality in training.

                lidar_np, object_bbx_center, object_bbx_mask = \
                self.augment(lidar_np, object_bbx_center, object_bbx_mask)

                lidar_dict = self.pre_processor.preprocess(lidar_np) # 点云单车预处理，生成体素特征，为三项，分别为体素特征(M, 5, 4)，体素坐标（M,3），体素实际点数(M,)
                selected_cav_processed.update({'processed_lidar': lidar_dict})

                # 直接做归一化，因为query-based方法直接对gt归一化目标做损失
                # object_bbx_center[:len(object_ids)] = self.encode_bbox(object_bbx_center[object_bbx_mask==1])   

            if self.visualize:
                selected_cav_processed.update({'origin_lidar': lidar_np})

            # camera
            if self.load_camera_file:
                # adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = [] # cam_to_lidar
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
                            "imgs": torch.stack(imgs), # [N, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            
            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center, # (100,) np array
                    "object_bbx_mask": object_bbx_mask, # (100,) np array
                    "object_ids": object_ids, # (N_gt,) list
                }
            )

            # query based方法为anchor free方法，因此在设计target时不需要anchor偏移计算
            # generate targets label
            # label_dict = self.post_processor.generate_label(
            #     gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            # )
            # selected_cav_processed.update({"label_dict": label_dict})

            return selected_cav_processed
            '''
            selected_cav_processed:{
                'processed_lidar':体素信息
                'object_bbx_center':
                'object_bbx_mask':
                'object_ids':
                'label_dict': 注意, 这里是single label
            }
            '''

        def collate_batch_train(self, batch):
            """
            Customized collate function for pytorch dataloader during training
            for early and late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # during training, we only care about ego.
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            processed_lidar_list = []
            # label_dict_list = []
            origin_lidar = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                # label_dict_list.append(ego_dict['label_dict'])
                
                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center)) # , dtype=np.float32
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
            # label_torch_dict = \
            #     self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            # label_torch_dict.update({'object_bbx_center': object_bbx_center,
            #                         'object_bbx_mask': object_bbx_mask})

            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'batch_size': len(batch)
                                    # 'anchor_box': torch.from_numpy(self.anchor_box),
                                    # 'label_dict': label_torch_dict
                                    })
            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.load_lidar_file:
                for i in range(len(batch)):
                    processed_lidar_list.append(batch[i]['ego']['processed_lidar']) # 预处理的体素信息
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(processed_lidar_list)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            if self.load_camera_file:
                # collate ego camera information
                imgs_batch = []
                rots_batch = []
                trans_batch = []
                intrins_batch = []
                extrinsics_batch = []
                post_trans_batch = []
                post_rots_batch = []
                for i in range(len(batch)):
                    ego_dict = batch[i]["ego"]["image_inputs"]
                    imgs_batch.append(ego_dict["imgs"])
                    rots_batch.append(ego_dict["rots"])
                    trans_batch.append(ego_dict["trans"])
                    intrins_batch.append(ego_dict["intrins"])
                    extrinsics_batch.append(ego_dict["extrinsics"])
                    post_trans_batch.append(ego_dict["post_trans"])
                    post_rots_batch.append(ego_dict["post_rots"])

                output_dict["ego"].update({
                    "image_inputs":
                        {
                            "imgs": torch.stack(imgs_batch),  # [B, N, C, H, W]
                            "rots": torch.stack(rots_batch),
                            "trans": torch.stack(trans_batch),
                            "intrins": torch.stack(intrins_batch),
                            "post_trans": torch.stack(post_trans_batch),
                            "post_rots": torch.stack(post_rots_batch),
                        }
                    }
                )


            return output_dict

        def collate_batch_test(self, batch):
            """
            Customized collate function for pytorch dataloader during testing
            for late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # currently, we only support batch size of 1 during testing
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            batch = batch[0]

            output_dict = {}

            # for late fusion, we also need to stack the lidar for better
            # visualization
            if self.visualize:
                projected_lidar_list = []
                origin_lidar = []

            for cav_id, cav_content in batch.items(): # keys are {'ego', 1}
                output_dict.update({cav_id: {}})
                # shape: (1, max_num, 7)
                object_bbx_center = \
                    torch.from_numpy(np.array([cav_content['object_bbx_center']]))
                object_bbx_mask = \
                    torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
                object_ids = cav_content['object_ids']

                # the anchor box is the same for all bounding boxes usually, thus
                # we don't need the batch dimension.
                # output_dict[cav_id].update(
                #     {"anchor_box": self.anchor_box_torch}
                # )

                transformation_matrix = cav_content['transformation_matrix']
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]
                    if (self.params['only_vis_ego'] is False) or (cav_id=='ego'):
                        projected_lidar = copy.deepcopy(cav_content['origin_lidar'])
                        projected_lidar[:, :3] = \
                            box_utils.project_points_by_matrix_torch(
                                projected_lidar[:, :3],
                                transformation_matrix)
                        projected_lidar_list.append(projected_lidar)

                if self.load_lidar_file:
                    # processed lidar dictionary
                    processed_lidar_torch_dict = \
                        self.pre_processor.collate_batch(
                            [cav_content['processed_lidar']]) # 体素先合到一起
                    output_dict[cav_id].update({'processed_lidar': processed_lidar_torch_dict})

                if self.load_camera_file:
                    imgs_batch = [cav_content["image_inputs"]["imgs"]]
                    rots_batch = [cav_content["image_inputs"]["rots"]]
                    trans_batch = [cav_content["image_inputs"]["trans"]]
                    intrins_batch = [cav_content["image_inputs"]["intrins"]]
                    extrinsics_batch = [cav_content["image_inputs"]["extrinsics"]]
                    post_trans_batch = [cav_content["image_inputs"]["post_trans"]]
                    post_rots_batch = [cav_content["image_inputs"]["post_rots"]]

                    output_dict[cav_id].update({
                        "image_inputs":
                            {
                                "imgs": torch.stack(imgs_batch),
                                "rots": torch.stack(rots_batch),
                                "trans": torch.stack(trans_batch),
                                "intrins": torch.stack(intrins_batch),
                                "extrinsics": torch.stack(extrinsics_batch),
                                "post_trans": torch.stack(post_trans_batch),
                                "post_rots": torch.stack(post_rots_batch),
                            }
                        }
                    )

                # label dictionary
                # label_torch_dict = \
                #     self.post_processor.collate_batch([cav_content['label_dict']]) # 这里是single标签
                    
                # for centerpoint
                # label_torch_dict.update({'object_bbx_center': object_bbx_center,
                #                          'object_bbx_mask': object_bbx_mask})

                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(
                        np.array(cav_content['transformation_matrix'])).float()
                
                # clean transformation matrix
                transformation_matrix_clean_torch = \
                    torch.from_numpy(
                        np.array(cav_content['transformation_matrix_clean'])).float()

                output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': object_bbx_mask,
                                            # 'label_dict': label_torch_dict,
                                            'object_ids': object_ids,
                                            'batch_size': 1,
                                            'transformation_matrix': transformation_matrix_torch,
                                            'transformation_matrix_clean': transformation_matrix_clean_torch})

                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar})

            if self.visualize:
                projected_lidar_stack = [torch.from_numpy(
                    np.vstack(projected_lidar_list))]
                output_dict['ego'].update({'origin_lidar': projected_lidar_stack})
                # output_dict['ego'].update({'projected_lidar_list': projected_lidar_list})
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
            pred_box_tensor, pred_score = self.post_processor.post_process_no_anchor(
                data_dict, output_dict
            )
            if self.dataset_name == 'dairv2x':
                gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict) # for dair-v2x datasets
            else:
                gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict) # for orther datasets
            
            return pred_box_tensor, pred_score, gt_box_tensor

        def post_process_no_fusion(self, data_dict, output_dict_ego):
            data_dict_ego = OrderedDict()
            data_dict_ego["ego"] = data_dict["ego"]
            if self.dataset_name == 'dairv2x':
                gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict) # for dair-v2x datasets
            else:
                gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict) # for orther datasets

            # pred_box_tensor, pred_score = self.post_processor.post_process(
            #     data_dict_ego, output_dict_ego
            # )
            # batch_size =  data_dict["ego"]['batch_size']
            # final_pred_dict = output_dict_ego['final_box_dicts']
            # print("output_dict_ego is ", output_dict_ego)
            pred_box_tensor = output_dict_ego["ego"]['pred_boxes']
            pred_score = output_dict_ego["ego"]['pred_scores']
            from opencood.utils import box_utils
            # 过滤
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
                                                        self.point_cloud_range,
                                                        order='lwh',
                                                        return_mask=True)
            pred_box_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box_tensor.device)
            pred_score = pred_score[mask]

            assert pred_score.shape[0] == pred_box_tensor.shape[0]

            return pred_box_tensor, pred_score, gt_box_tensor

        def post_process_no_fusion_uncertainty(self, data_dict, output_dict_ego):
            data_dict_ego = OrderedDict()
            data_dict_ego['ego'] = data_dict['ego']
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            pred_box_tensor, pred_score, cls_noise, uncertainty, unc_epi_cls, unc_epi_reg = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, cls_noise, uncertainty, unc_epi_cls, unc_epi_reg

        # def post_process_query()

    return LateV2FusionDataset
