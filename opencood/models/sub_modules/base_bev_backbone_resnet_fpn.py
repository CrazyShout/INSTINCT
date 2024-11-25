"""
Resblock is much strong than normal conv

Provide api for multiscale intermeidate fuion
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock

DEBUG = False

class ResNetBEVBackboneFPN(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_dropout = model_cfg.get('use_dropout', False)
        self.enable_dropout = model_cfg.get('dropout_enable', False)
        if self.use_dropout:
            print("===backbone use dropout===")
            if self.enable_dropout:
                print("  --enforce enable dropout with F.Dropout2d")

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums'] # [3, 4, 5]
            layer_strides = self.model_cfg['layer_strides'] # [2, 2, 2]
            num_filters = self.model_cfg['num_filters'] # [128, 256, 512]
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter'] # [128, 128, 128]
            upsample_strides = self.model_cfg['upsample_strides'] # [1, 2, 4]

        else:
            upsample_strides = num_upsample_filters = []

        self.resnet = ResNetModified(BasicBlock, 
                                        layer_nums,
                                        layer_strides,
                                        num_filters,
                                        inplanes = model_cfg.get('inplanes', 64))

        num_levels = len(layer_nums)
        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()

        self.lateral_convs = nn.ModuleList()
        for idx in range(num_levels):
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(num_filters[idx], num_upsample_filters[idx], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))

        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.lateral_conv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
                                        nn.BatchNorm2d(128, eps=1e-3,momentum=0.01),
                                        nn.ReLU())

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        x = self.resnet(spatial_features)  # tuple of features
        ups = []

        for i in range(self.num_levels-1, -1, -1):
            if len(self.deblocks) > 0:
                if i == 2: # 最底层
                    temp = self.deblocks[i](x[i]) # 空间2倍上采样，通道2倍下采样
                    res = self.lateral_conv1(temp) # 空间2倍上采样， 通道2倍下采样
                elif i == 1:
                    temp = temp + x[i] # 非就地操作，先加上
                    temp = self.deblocks[i](temp) # 空间2倍上采样， 通道2倍下采样
                    res = temp
                elif i == 0:
                    temp = temp + x[i]
                    res = self.deblocks[i](temp) # 空间分辨率不变，通道分辨率不变
                else:
                    raise ValueError("here cause a big issue!")
                
                if self.use_dropout:
                    if self.enable_dropout: # 这个开启的时候则在验证和推理的时候也会开启Dropout
                        ups.append(F.dropout2d(res, p=0.1, training = True))
                    else:
                        ups.append(F.dropout2d(res, p=0.1, training = self.training))
                else:
                    ups.append(res)
            else:
                ups.append(x[i])

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict

    # these two functions are seperated for multiscale intermediate fusion
    def get_multiscale_feature(self, spatial_features):
        """
        before multiscale intermediate fusion
        """
        x = self.resnet(spatial_features)  # tuple of features
        return x

    def decode_multiscale_feature(self, x):
        """
        after multiscale interemediate fusion
        """
        ups = []
        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        return x
        
    def get_layer_i_feature(self, spatial_features, layer_i):
        """
        before multiscale intermediate fusion
        """
        return eval(f"self.resnet.layer{layer_i}")(spatial_features)  # tuple of features
    