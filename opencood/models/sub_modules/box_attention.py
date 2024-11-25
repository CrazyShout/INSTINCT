import math

import torch
import torch.nn.functional as F
from torch import nn

from opencood.pcdet_utils.box_attention.box_attention_func import BoxAttnFunction


class Box3dAttention(nn.Module):
    def __init__(self, d_model, num_level, num_head, with_rotation=True, kernel_size=5):
        super(Box3dAttention, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"

        num_variable = 5 if with_rotation else 4

        self.im2col_step = 64
        self.d_model = d_model # 256
        self.num_head = num_head # 8
        self.num_level = num_level # 1
        self.head_dim = d_model // num_head # 32
        self.with_rotation = with_rotation # query选择的时候使用的False， Decoder中使用的True
        self.num_variable = num_variable # 4 or 5
        self.kernel_size = kernel_size # 5
        self.num_point = kernel_size ** 2 # 采样点个数

        self.linear_box = nn.Linear(d_model, num_level * num_head * num_variable)

        self.linear_attn = nn.Linear(d_model, num_head * num_level * self.num_point) # 256--> 8*1*25


        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2
            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2
            indices = torch.linspace(start_idx, end_idx, kernel_size) # -2 到 2 生成5个点的坐标
        i, j = torch.meshgrid(indices, indices) # 两个5 * 5 的矩阵，表示i，j 坐标
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / kernel_size # （5x5， 2）并归一化
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        
        nn.init.constant_(self.linear_attn.weight, 0.0)
        nn.init.constant_(self.linear_attn.bias, 0.0)
        nn.init.constant_(self.linear_box.weight, 0.0)
        nn.init.uniform_(self.linear_box.bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows, h_size=188.0):
        B, L = ref_windows.shape[:2]

        offset_boxes = self.linear_box(query)
        offset_boxes = offset_boxes.view(B, L, self.num_head, self.num_level, self.num_variable)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        ref_boxes = ref_windows[..., [0, 1, 3, 4]]
        ref_angles = ref_windows[..., [6]]

        if self.with_rotation:
            offset_boxes, offset_angles = offset_boxes.split(4, dim=-1)
            angles = (ref_angles + offset_angles / 16) * 2 * math.pi
        else:
            angles = ref_angles.expand(B, L, self.num_head, self.num_level, 1)

        boxes = ref_boxes + offset_boxes / 8 * ref_boxes[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        rot_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
        rot_matrix = rot_matrix.view(B, L, self.num_head, self.num_level, 1, 2, 2)

        grid = self.kernel_indices * torch.relu(size)
        grid = center + (grid.unsqueeze(-2) * rot_matrix).sum(-1)

        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows):
        '''
            query: self.with_pos_embed(select_src, select_pos), # (B, foreground_num, 256)  加上位置编码的前景特征
            value: src, #  (B, H*W, 256)
            v_shape: src_shape, # (1, 2) 记录着特征图形状
            v_mask: None,
            v_start_index: src_start_idx, # 划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)
            v_valid_ratios: None,
            ref_windows: select_ref_windows, # (B, foreground_num, 7) 参考窗口 BoxAttention需要的
        '''
        B, LQ = query.shape[:2] # LQ 是前景query数量
        LV = value.shape[1] # LV 是总query数量 == H*W

        value = self.value_proj(value) # 线性变换 维度不变
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(B, LV, self.num_head, self.head_dim) # (B, H*W, 8，32)

        attn_weights = self.linear_attn(query) # 线性变换  (B, foreground_num, 256) --> (B, foreground_num, 8*1*25)
        attn_weights = F.softmax(attn_weights.view(B, LQ, self.num_head, -1), dim=-1) # (B, foreground_num, 8, 1*25) 最后一维度求softmax
        attn_weights = attn_weights.view(B, LQ, self.num_head, self.num_level, self.kernel_size, self.kernel_size) # (B, foreground_num, 8, 1, 5, 5) # 这个应该是25个采样点的权重

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows, h_size=LV**(0.5)) # 确定采样网格 (B, foreground_num, 8， 1，5x5，2) 这会在后面对网格中标记的位置进行采样

        output = BoxAttnFunction.apply(value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step) # (B, foreground_num, 32*8) # 采样对应数量的前景特征，并对每个特征进行25个采样点加权求和
        output = self.out_proj(output) # (B, foreground_num, 256)

        return output, attn_weights

