# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch
from cumm import tensorview as tv
from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        # preprocess_cfg：yaml中preprocess项，在父类中被赋予给self.params
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            # spconv v2.x
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
            self.spconv = 2
        self.lidar_range = self.params['cav_lidar_range'] # 雷达范围 [-100.8, -40, -3.5, 100.8, 40, 1.5]
        self.voxel_size = self.params['args']['voxel_size'] # 体素大小 [0.1, 0.1, 0.1]
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel'] # 每个体素中最大的点数 5

        if train: # 训练时考虑的体素数很少
            self.max_voxels = self.params['args']['max_voxel_train'] # 最大体素数为32000
        else:
            self.max_voxels = self.params['args']['max_voxel_test'] # 最大体素为70000

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64) # 这里计算了网格大小并四舍五入，但是有疑问：加载YAML文件时​load_point_pillar_params方法中已经求出且就放在self.param中，为何又要计算？答：可能是其他eval加载方法中没有计算，这里做个兼容？

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
        else:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels
            )

    def preprocess(self, pcd_np):
        data_dict = {}
        if self.spconv == 1:
            voxel_output = self.voxel_generator.generate(pcd_np)
        else:
            pcd_tv = tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv) # 返回字典，分别为体素信息，体素在三维网格的坐标，每个体素中实际点数
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy() # （M, 5, 4）
            coordinates = coordinates.numpy() # (M, 3)
            num_points = num_points.numpy() # (M,)

        data_dict['voxel_features'] = voxels # 体素的个数 [M, 5, 4]
        data_dict['voxel_coords'] = coordinates # 体素的坐标 [M, 3]
        data_dict['voxel_num_points'] = num_points # 每个体素中的点的数量

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features']) # （M, 5, 4）
            voxel_num_points.append(batch[i]['voxel_num_points']) # (M, )
            coords = batch[i]['voxel_coords'] # (M, 3)
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i)) # 填充batch id 变为(M, 4)

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points)) # (N_b, )
        voxel_features = torch.from_numpy(np.concatenate(voxel_features)) # (N_b, 5, 4)
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords)) # (N_b, 4)

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features'])) # 连接第一维 eg. [[3000, 32, 4],[2000, 32, 4]] -> [5000, 32, 4]
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points'])) # eg. [[3000], [2000]] -> [5000]
        coords = batch['voxel_coords'] # [[3000, 3], [2000, 3]] 列表长度为协同车的个数
        voxel_coords = []
        # print("len(coords) is ", len(coords))
        for i in range(len(coords)): # 遍历一个batch中的所有的体素坐标
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i)) # 逐个遍历坐标，在第二个维度（列）的最前面加上编号 范围0~K-1， K为coords的长度也就是整个batch中所有样本的协同车数量之和
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords)) # [[3000, 4],[2000, 4]] -> [5000, 4]

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}
