from functools import partial
import torch
import torch.nn as nn

from opencood.pcdet_utils.spconv_utils import replace_feature, spconv
from spconv.pytorch import ConvAlgo

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, algo = ConvAlgo.Native):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key, algo = ConvAlgo.MaskImplicitGemm)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelResBackBone8xVoxelNeXt(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('spconv_kernel_sizes', [3, 3, 3, 3]) # [5, 3, 3, 3]
        channels = model_cfg.get('channels', [16, 32, 64, 128, 128]) # [32, 64, 128, 256, 256]
        out_channel = model_cfg.get('out_channel', 128) # 256

        self.sparse_shape = grid_size[::-1] + [1, 0, 0] # 51, 800, 2016

        # -->(51, 800, 2016) C=32
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        ) 
        block = post_act_block

        # -->(51, 800, 2016) C=32
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        # -->(26, 400, 1008) C=64
        self.conv2 = spconv.SparseSequential(
            # spconv.SubMConv3d(channels[0], channels[1], 3, stride=2,padding=2, bias=False, indice_key='subm1'),
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        # -->(13, 200, 504) C=128
        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        # -->(7, 100, 252) C=256
        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        # -->(4, 50, 126) C=256
        self.conv5 = spconv.SparseSequential(
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        
        # -->(2, 25, 63) C=256
        self.conv6 = spconv.SparseSequential(
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )

        # -->(100, 252) C=256 第4/5/6层多尺度合并为bev后使用稀疏2D卷积调整feature， 注意，子流形一般适合用来详细提取信息，而稀疏卷积一般额外用来下采样
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

    def bev_out(self, x_conv):
        features_cat = x_conv.features # n4+n5+n6, c
        indices_cat = x_conv.indices[:, [0, 2, 3]] # n4+n5+n6, 3 丢弃z轴
        spatial_shape = x_conv.spatial_shape[1:] # H，W

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True) # 第一个是唯一的元素（M，3），第二个是每一个唯一的元素在原始数据中的逆索引（n4+n5+n6，）
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1])) # 形成M，C张量 M是唯一元素个数
        features_unique.index_add_(0, _inv, features_cat) # 这是在将相同的空间位置的元素加在一起

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size'] # 这里是总共有多少个特征图
        # print("voxel_features shape is:", voxel_features.shape)
        # print("voxel_coords shape is:", voxel_coords.shape)
        # print("batch_size is:", batch_size)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features, # n,c
            indices=voxel_coords.int(), # n, 4
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2 # 稀疏张量对象有两个部分，一个是features：n，c 还有就是 indices：n, 4，因此放大索引就可以在空间上放大
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features])) # n4+n5+n6, c
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out, # （256， 100， 256）但还是稀疏张量 需要转变成 dense tensor
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict


class VoxelResBackBone8xLight(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes = model_cfg.get('spconv_kernel_sizes', [3, 3, 3, 3]) # [5, 3, 3, 3]
        channels = model_cfg.get('channels', [16, 32, 64, 128, 128]) # [32, 64, 128, 256, 256]
        out_channel = model_cfg.get('out_channel', 128) # 256

        self.sparse_shape = grid_size[::-1] + [1, 0, 0] # 51, 800, 2016

        # -->(51, 800, 2016) C=32
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        ) 
        block = post_act_block

        # -->(51, 800, 2016) C=32
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        # -->(26, 400, 1008) C=64
        self.conv2 = spconv.SparseSequential(
            # spconv.SubMConv3d(channels[0], channels[1], 3, stride=2,padding=2, bias=False, indice_key='subm1'),
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        # -->(13, 200, 504) C=128
        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        # -->(7, 100, 252) C=256
        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        # -->(4, 50, 126) C=256
        self.conv5 = spconv.SparseSequential(
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        
        # -->(2, 25, 63) C=256
        self.conv6 = spconv.SparseSequential(
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )

        # -->(100, 252) C=256 第4/5/6层多尺度合并为bev后使用稀疏2D卷积调整feature， 注意，子流形一般适合用来详细提取信息，而稀疏卷积一般额外用来下采样
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

    def bev_out(self, x_conv):
        features_cat = x_conv.features # n4+n5+n6, c
        indices_cat = x_conv.indices[:, [0, 2, 3]] # n4+n5+n6, 3 丢弃z轴
        spatial_shape = x_conv.spatial_shape[1:] # H，W

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True) # 第一个是唯一的元素（M，3），第二个是每一个唯一的元素在原始数据中的逆索引（n4+n5+n6，）
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1])) # 形成M，C张量 M是唯一元素个数
        features_unique.index_add_(0, _inv, features_cat) # 这是在将相同的空间位置的元素加在一起

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size'] # 这里是总共有多少个特征图
        # print("voxel_features shape is:", voxel_features.shape)
        # print("voxel_coords shape is:", voxel_coords.shape)
        # print("batch_size is:", batch_size)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features, # n,c
            indices=voxel_coords.int(), # n, 4
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2 # 稀疏张量对象有两个部分，一个是features：n，c 还有就是 indices：n, 4，因此放大索引就可以在空间上放大
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features])) # n4+n5+n6, c
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out, # （256， 100， 256）但还是稀疏张量 需要转变成 dense tensor
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict