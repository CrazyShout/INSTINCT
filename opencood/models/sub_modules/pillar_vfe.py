"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part: # （M, 32, 10）
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0) # 分组进行，然后拼起来 M’， 32， 10
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x # permute是因为bn操作对输入的shape要求
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0] # （M, 1, 64）

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm'] # 基本默认是True
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz'] # 基本默认是True
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters'] # 64
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) # [4+6, 64] 

        pfn_layers = []
        for i in range(len(num_filters) - 1): # 只会运行一次
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] # 体素偏移位置，在左上角体素中心
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0): # 每个体素中实际点个数，体素中最多的点个数
        actual_num = torch.unsqueeze(actual_num, axis + 1) # （M, 1）
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
        max_num_shape[axis + 1] = -1 # [1, -1]
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape) # (1, 32) 从0到31
        paddings_indicator = actual_num.int() > max_num # (M, 32) 得到每个体素中的点数掩码，能识别0到32个，超过32个默认掩掉
        return paddings_indicator

    def forward(self, batch_dict):
        """encoding voxel feature using point-pillar method
        Args:
            voxel_features: [M, 32, 4]
            voxel_num_points: [M,]
            voxel_coords: [M, 4]
        Returns:
            features: [M,64], after PFN
        """
        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']

        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1) # 求出每一个pillar里的平均值 （M, 1, 3）
        f_cluster = voxel_features[:, :, :3] - points_mean # （M, 32, 3）求出每个点与均值的偏差 

        f_center = torch.zeros_like(voxel_features[:, :, :3]) # （M, 32, 3）
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset) # 每个点的实际x坐标-每个体素的实际x坐标（体素中心） 
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center] # 4 + 3+ 3
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) # （M， 32， 10）

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0) # (M, 32)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) # (M, 32， 1)
        features *= mask # （M， 32， 10） #  这是将补0的点，对其所有的10维度全部清零
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze() # (M, 1, 64) --> (M, 64)
        batch_dict['pillar_features'] = features

        return batch_dict
