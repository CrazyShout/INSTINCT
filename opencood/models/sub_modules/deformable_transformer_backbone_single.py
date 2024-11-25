import numpy as np
import torch
import math
import torch.nn as nn
from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock, Bottleneck
from opencood.models.sub_modules.detr_module import PositionEmbeddingSine, \
                DeformableTransformerEncoderLayer, DeformableTransformerEncoder
# from opencood.models.fuse_modules.self_attn import AttFusion
# from opencood.models.fuse_modules.deform_fuse import DeformFusion
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


DEBUG = False


"""
    Different from MaxFusion in max_fuse.py
    This is a simplified version.
    pairwise_t_matrix is already scaled.
"""
def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class MaxFusion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, record_len, pairwise_t_matrix):
        """
        pairwise_t_matrix is already normalized [B, L, L, 2, 3]
        """
        split_x = regroup(x, record_len) # List[]
        batch_size = len(record_len)
        C, H, W = split_x[0].shape[1:]  # C, W, H before
        out = []
        for b, xx in enumerate(split_x):
            N = xx.shape[0]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0
            xx = warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W))

            h = torch.max(xx, dim=0)[0]  # C, W, H before
            out.append(h)
        return torch.stack(out, dim=0)

class DeformableTransformerBackbone(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        
        self.proj_first = True
        if ('proj_first' in model_cfg) and (model_cfg['proj_first'] is False):
            self.proj_first = False
            self.discrete_ratio = model_cfg['voxel_size'][0]
            self.downsample_rate = 1

        self.level_num = len(model_cfg['layer_nums']) # exactly 3 now

        layer_nums = model_cfg['layer_nums'] #  [3, 4, 5]
        num_filters = model_cfg['num_filters'] # [64, 128, 256]
        layer_strides = model_cfg['layer_strides'] # [2, 2, 2]
        hidden_dim = model_cfg['hidden_dim'] # 256
        upsample_strides = model_cfg['upsample_strides'] # [1, 2, 4]
        num_upsample_filters = model_cfg['num_upsample_filter'] # [128, 128, 128]

        self.resnet = ResNetModified(BasicBlock, 
                                        layer_nums,
                                        layer_strides,
                                        num_filters)

        self.position_embedding = PositionEmbeddingSine(hidden_dim//2)

        self.hidden_dim = hidden_dim

        input_proj_list = []
        for i in range(self.level_num):
            proj_in_channels = num_filters[i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(proj_in_channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))

        self.input_proj = nn.ModuleList(input_proj_list)
        self.level_embed = nn.Parameter(torch.Tensor(self.level_num, self.hidden_dim))
        self.upsample_strides = model_cfg['upsample_strides']

        encoder_layer = DeformableTransformerEncoderLayer(self.hidden_dim, model_cfg['dim_feedforward'],
                                                          model_cfg['dropout'], model_cfg['activation'],
                                                          self.level_num, model_cfg['n_head'], model_cfg['enc_n_points'])
        self.encoder = DeformableTransformerEncoder(encoder_layer, model_cfg['num_encoder_layers'])

        self.deblocks = nn.ModuleList()
        for idx in range(self.level_num):
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    self.hidden_dim, num_upsample_filters[idx],
                    upsample_strides[idx],
                    stride=upsample_strides[idx], bias=False
                ),
                nn.BatchNorm2d(num_upsample_filters[idx],
                                eps=1e-3, momentum=0.01),
                nn.ReLU()
                ))

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features'] # (B, C, H, W)
        if DEBUG:
            origin_features = torch.clone(spatial_features)

        batch_size = data_dict['batch_size'] # [n1, n2]

        ups = []
        ret_dict = {}
        x = spatial_features

        # B = len(record_len)
        H, W = x.shape[2:]  ## this is original feature map [200, 704], not downsampled

        features = self.resnet(x)  # feature[i] is (sum(cav), C, H, W), different i, different C, H, W

        # 单车的时候不需要fuse 所以直接就是多尺度特征
        x_fuseds = features # [(B, C, H/2, W/2), (B, 2C, H/4, W/4), (B, 4C, H/8, W/8)]

        pos_embeds = [self.position_embedding(x_fused) for x_fused in x_fuseds] # 和下面相同形状
        srcs = [self.input_proj[i](x_fuseds[i]) for i in range(len(x_fuseds))] # [(B, 256，H/2, W/2), (B, 256，H/4, W/4), (B, 256，H/8, W/8)]

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # bs, c, hw -> bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # bs, hw, c
            
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed) # bs, hw, c
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1) # (bs, l_all, c)
        mask_flatten = torch.zeros(src_flatten.shape[:2], device=src_flatten.device, dtype=torch.bool)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # (3, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [0, l1_num, l1+l2 num]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in srcs], 1)


        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        flatten_length = [h*w for (h,w) in spatial_shapes]
        output_split = torch.split(memory, flatten_length, dim=1)
        output_features = [output.reshape(bs,spatial_shapes[i][0], spatial_shapes[i][1],self.hidden_dim).permute(0,3,1,2) for i, output in enumerate(output_split)]
        
        ups = []
        for i, feat in enumerate(output_features):
            feat = self.deblocks[i](feat)
            ups.append(feat)

        ups = torch.cat(ups, dim=1)

        x = ups

        data_dict['spatial_features_2d'] = x
        return data_dict


    def get_valid_ratio(self, x):
        N, _, H, W = x.shape
        mask = torch.zeros((N,H,W),dtype=torch.bool,device=x.device)
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


# class DeformableTransformerBackbone(nn.Module):
#     def __init__(self, model_cfg, input_channels):
#         super().__init__()
#         self.model_cfg = model_cfg
#         self.compress = False

        
#         self.proj_first = True
#         if ('proj_first' in model_cfg) and (model_cfg['proj_first'] is False):
#             self.proj_first = False
#             self.discrete_ratio = model_cfg['voxel_size'][0]
#             self.downsample_rate = 1

#         self.level_num = len(model_cfg['layer_nums']) # exactly 3 now

#         layer_nums = model_cfg['layer_nums'] #  [3, 4, 5]
#         num_filters = model_cfg['num_filters'] # [64, 128, 256]
#         layer_strides = model_cfg['layer_strides'] # [2, 2, 2]
#         hidden_dim = model_cfg['hidden_dim'] # 256
#         upsample_strides = model_cfg['upsample_strides'] # [1, 2, 4]
#         num_upsample_filters = model_cfg['num_upsample_filter'] # [128, 128, 128]

#         self.pos_embed = PositionEmbeddingSine(self.hidden_dim // 2)

#         self.hidden_dim = hidden_dim

#         self.input_proj = nn.Sequential(
#             nn.Conv2d(input_channels, self.hidden_dim, kernel_size=1),
#             nn.GroupNorm(32, self.hidden_dim),
#         )
#         self.upsample_strides = model_cfg['upsample_strides']

#         encoder_layer = DeformableTransformerEncoderLayer(self.hidden_dim, model_cfg['dim_feedforward'],
#                                                           model_cfg['dropout'], model_cfg['activation'],
#                                                           self.level_num, model_cfg['n_head'], model_cfg['enc_n_points'])
#         self.encoder = DeformableTransformerEncoder(encoder_layer, model_cfg['num_encoder_layers'])

#     def forward(self, data_dict):
#         spatial_features_2d = data_dict['spatial_features_2d'] # (B, C, H, W)
#         if DEBUG:
#             origin_features = torch.clone(spatial_features_2d)

#         batch_size = data_dict['batch_size'] # [n1, n2]


#         ups = []
#         ret_dict = {}
#         x = spatial_features_2d

#         H, W = x.shape[2:]  ## this is original feature map [200, 704], not downsampled
#         features = []
#         pos_encodings = []
#         feature = self.input_proj(spatial_features_2d)
#         feature = feature.flatten(2).transpose(1, 2) # (B, HW, C)
#         pos_encoding = self.pos_embed(spatial_features_2d) # 位置编码 B，256， H， W
#         pos_encoding = pos_encoding.flatten(2).transpose(1, 2) # (B, HW, C)
#         spatial_shape = torch.as_tensor([(H, W)], dtype=torch.long, device=pos_encoding.device) # (1, 2)
#         level_start_index = torch.cat([spatial_shape.new_zeros(1), spatial_shape.prod(1).cumsum(0)[:-1]])

#         valid_ratios = torch.stack([self.get_valid_ratio(feature)], 1)

#         mask_flatten = torch.zeros(feature.shape[:2], device=feature.device, dtype=torch.bool)

#         memory = self.encoder(feature, spatial_shape, level_start_index, valid_ratios, pos_encoding, mask_flatten) # (B, HW, 256)

#         flatten_length = [h*w for (h,w) in spatial_shape]
#         output_split = torch.split(memory, flatten_length, dim=1)
#         output_features = [output.reshape(batch_size,spatial_shape[i][0], spatial_shape[i][1],self.hidden_dim).permute(0,3,1,2) for i, output in enumerate(output_split)]


#         features.append(self.input_proj(spatial_features_2d))
#         pos_encodings.append(self.pos_embed(spatial_features_2d)) # 位置编码 B，256， H， W

#         src_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
#             bs, c, h, w = src.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             src = src.flatten(2).transpose(1, 2) # bs, c, hw -> bs, hw, c
#             pos_embed = pos_embed.flatten(2).transpose(1, 2) # bs, hw, c
            
#             lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
#             lvl_pos_embed_flatten.append(lvl_pos_embed) # bs, hw, c
#             src_flatten.append(src)
#         src_flatten = torch.cat(src_flatten, 1) # (bs, l_all, c)
#         mask_flatten = torch.zeros(src_flatten.shape[:2], device=src_flatten.device, dtype=torch.bool)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # (3, 2)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [0, l1_num, l1+l2 num]
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in srcs], 1)


#         memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
#         flatten_length = [h*w for (h,w) in spatial_shapes]
#         output_split = torch.split(memory, flatten_length, dim=1)
#         output_features = [output.reshape(bs,spatial_shapes[i][0], spatial_shapes[i][1],self.hidden_dim).permute(0,3,1,2) for i, output in enumerate(output_split)]
        
#         ups = []
#         for i, feat in enumerate(output_features):
#             feat = self.deblocks[i](feat)
#             ups.append(feat)

#         ups = torch.cat(ups, dim=1)

#         x = ups

#         data_dict['spatial_features_2d'] = x
#         return data_dict


#     def get_valid_ratio(self, x):
#         N, _, H, W = x.shape
#         mask = torch.zeros((N,H,W),dtype=torch.bool,device=x.device)
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio
