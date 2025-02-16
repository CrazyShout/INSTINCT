import copy

import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_utils
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils
from .target_assigner.hungarian_assigner_qs import HungarianMatcher3d, generalized_box3d_iou, \
    box_cxcyczlwh_to_xyxyxy
from opencood.models.sub_modules.cdn import prepare_for_cdn, dn_post_process_w_ious
from opencood.models.sub_modules.ConQueR_transformer import Transformer, MLP, get_clones


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats # 128
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is not None:
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            size_h, size_w = x.shape[-2:]
            y_embed = torch.arange(1, size_h + 1, dtype=x.dtype, device=x.device)
            x_embed = torch.arange(1, size_w + 1, dtype=x.dtype, device=x.device)
            y_embed, x_embed = torch.meshgrid(y_embed, x_embed)
            x_embed = x_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)
            y_embed = y_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * dim_t.div(2, rounding_mode="floor") / self.num_pos_feats) # (128, )

        pos_x = x_embed[:, :, :, None] / dim_t # b, h, w, 1 除以 频率缩放后 为 b, h, w, 128
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # b, h, w, 128
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class Det3DHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=1, code_size=7, num_layers=1):
        super().__init__()
        class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3) # (B, L, 3)
        bbox_embed = MLP(hidden_dim, hidden_dim, code_size, 3) # (B, L, 7)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.layers[-1].bias.data = torch.ones(num_classes) * bias_value # 最后一个线性层的偏置进行初始化
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

        self.class_embed = get_clones(class_embed, num_layers)
        self.bbox_embed = get_clones(bbox_embed, num_layers)

        iou_embed = MLP(hidden_dim, hidden_dim, 1, 3) # 每个预测框的IoU估计值
        nn.init.constant_(iou_embed.layers[-1].weight.data, 0)
        nn.init.constant_(iou_embed.layers[-1].bias.data, 0)
        self.iou_embed = get_clones(iou_embed, num_layers)

    def forward(self, embed, anchors, layer_idx=0):
        cls_logits = self.class_embed[layer_idx](embed)
        box_coords = (self.bbox_embed[layer_idx](embed) + inverse_sigmoid(anchors)).sigmoid() # 这里类似锚框的机制，逆转回实数空间加上预测的偏移，最后重新归一化
        pred_iou = (self.iou_embed[layer_idx](embed)).clamp(-1, 1)
        return cls_logits, box_coords, pred_iou


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super(MaskPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.reshape([b, c, -1]).permute(0, 2, 1)
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.hidden_dim // 2, dim=-1) # 分成两部分 都是 b,hxw, 128
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1) # 全局的均值 形状不变
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z) # b, hxw, 1
        out = out.reshape([b, h, w])
        return out


class ConQueRHead(nn.Module):
    def __init__(
            self,
            model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
            predict_boxes_when_training=True, train_flag=True
    ):
        super(ConQueRHead, self).__init__()

        self.grid_size = grid_size # [2016, 800, 50]
        self.point_cloud_range = point_cloud_range # [-75.2, -75.2, -2, 75.2, 75.2, 4]
        self.voxel_size = voxel_size # [0.1, 0.1, 0.15]
        self.feature_map_stride = model_cfg['feature_map_stride'] # 8
        self.num_classes = num_class # 1

        self.model_cfg = model_cfg
        self.train_flag = train_flag

        self.hidden_channel = self.model_cfg['hidden_channel'] # 256
        self.num_queries = self.model_cfg['num_queries'] # 1000
        self.aux_loss = self.model_cfg['loss_config']['aux_loss'] # True
        self.keep_ratio = self.model_cfg['keep_ratio'] # 0.3
        self.iou_cls = self.model_cfg['iou_cls'] # [0]
        self.iou_rectifier = self.model_cfg['iou_rectifier'] # 0.68

        num_heads = self.model_cfg['num_heads'] # 8
        dropout = self.model_cfg['dropout'] # 0.0
        activation = self.model_cfg['activation']
        ffn_channel = self.model_cfg['ffn_channel'] # 1024
        num_decoder_layers = self.model_cfg['num_decoder_layers'] # 6
        self.code_size = self.model_cfg['code_size'] # 7

        cp_flag = self.model_cfg['cp'] # True

        self.dn = self.model_cfg['dn']

        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_channel, kernel_size=1),
            nn.GroupNorm(32, self.hidden_channel),
        )

        self.pos_embed = PositionEmbeddingSine(self.hidden_channel // 2)

        for module in self.input_proj.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # self.mask_predictor = MaskPredictor(self.hidden_channel)

        self.transformer = Transformer(
            d_model=self.hidden_channel, # 256
            nhead=num_heads, # 8
            nlevel=1,
            num_encoder_layers=num_decoder_layers, # 6
            num_decoder_layers=num_decoder_layers, # 6
            dim_feedforward=ffn_channel, # 1024
            dropout=dropout, # 0.0
            activation=activation, # gelu
            num_queries=self.num_queries, # 1000
            num_classes=num_class, # 3
            cp_flag=cp_flag # True
        )

        self.transformer.proposal_head = Det3DHead(
            self.hidden_channel,
            code_size=self.code_size,
            num_classes=num_class,
            num_layers=1,
        )
        self.transformer.decoder.detection_head = Det3DHead(
            self.hidden_channel,
            code_size=self.code_size,
            num_classes=num_class,
            num_layers=num_decoder_layers,
        )

        if self.train_flag and self.dn['enabled']:
            contras_dim = self.model_cfg['contrastive']['dim'] # 256
            self.eqco = self.model_cfg['contrastive']['eqco'] # 1000
            self.tau = self.model_cfg['contrastive']['tau'] # 0.7
            self.contras_loss_coeff = self.model_cfg['contrastive']['loss_coeff'] # 0.2
            self.projector = nn.Sequential(
                nn.Linear(self.code_size + self.num_classes, contras_dim),
                nn.ReLU(),
                nn.Linear(contras_dim, contras_dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(contras_dim, contras_dim),
                nn.ReLU(),
                nn.Linear(contras_dim, contras_dim),
            )
            self.similarity_f = nn.CosineSimilarity(dim=2)

            self.transformer.decoder_gt = copy.deepcopy(self.transformer.decoder)
            for param_q, param_k in zip(self.transformer.decoder.parameters(),
                                        self.transformer.decoder_gt.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient


        self.assigner = HungarianMatcher3d(
            cost_class=self.model_cfg['target_assigner_config']['hungarian_assigner']['cls_cost'],   # 1.0
            cost_bbox=self.model_cfg['target_assigner_config']['hungarian_assigner']['bbox_cost'],   # 4.0
            cost_giou=self.model_cfg['target_assigner_config']['hungarian_assigner']['iou_cost'],    # 2.0
            cost_rad=self.model_cfg['target_assigner_config']['hungarian_assigner']['rad_cost'],     # 4.0
            iou_rectifier=self.iou_rectifier,
            iou_cls=self.iou_cls
        )

        weight_dict = {
            "loss_ce": self.model_cfg['target_assigner_config']['hungarian_assigner']['cls_cost'],
            "loss_bbox": self.model_cfg['target_assigner_config']['hungarian_assigner']['bbox_cost'],
            "loss_giou": self.model_cfg['target_assigner_config']['hungarian_assigner']['iou_cost'],
            "loss_rad": self.model_cfg['target_assigner_config']['hungarian_assigner']['rad_cost'],
        }
        losses = ["focal_labels", "boxes"]
        self.losses = Det3DLoss(
            matcher=self.assigner,
            weight_dict=weight_dict,
            losses=losses,
            decode_func=self.decode_bbox,
            aux_loss=self.aux_loss
        )

        # setup aux loss weight
        if self.aux_loss:
            aux_weight_dict = {}
            if hasattr(self.losses, "weight_dict"):
                aux_weight_dict.update({k + f"_dn": v for k, v in self.losses.weight_dict.items()}) # 增加了{'loss_ce_dn': 1.0, ...} 是给dn使用的
                for i in range(num_decoder_layers - 1): # 设置5层的dn损失的权重
                    aux_weight_dict.update({k + f"_dn_{i}": v for k, v in self.losses.weight_dict.items()})
                    aux_weight_dict.update({k + f"_{i}": v for k, v in self.losses.weight_dict.items()})
                self.losses.weight_dict.update(aux_weight_dict)

    def predict(self, batch_dict):
        batch_size = batch_dict['batch_size']
        spatial_features_2d = batch_dict['spatial_features_2d'] # B, 512, H, W

        features = []
        pos_encodings = []
        features.append(self.input_proj(spatial_features_2d)) # 放入一个B，256，H，W
        pos_encodings.append(self.pos_embed(spatial_features_2d)) # 位置编码 B，256， H， W

        # score_mask = self.mask_predictor(features[0]) # B, H, W

        dn = self.dn
        if self.train_flag and dn['enabled'] and dn['dn_number'] > 0:
            gt_boxes = batch_dict['object_bbx_center'] # B, maxnum, 7
            gt_boxes_mask = batch_dict['object_bbx_mask'] # B, maxnum

            targets = list() # 列表存放每个样本的标签
            for batch_idx in range(batch_size):
                target = {}
                gt_bboxes = gt_boxes[batch_idx] # (maxnum, 7)
                gt_bboxes_mask = gt_boxes_mask[batch_idx] # (maxnum, )
                valid_box = gt_bboxes[gt_bboxes_mask.bool()] # （n_idx, 7）
                gt_labels = torch.ones(valid_box.size(0), device=valid_box.device, dtype=valid_box.dtype)
                target['gt_boxes'] = self.encode_bbox(valid_box) # 给gt box做好归一化工作
                target['labels'] = gt_labels.long() - 1 # (n_idx, )
                targets.append(target)

            '''
            若targets中每个样本的gt objet 个数为 n1, n2 其中 n2 最大 记作 max_gt_num
            input_query_label: (B, pad_size, num_classes) num_classes=1 其实这项没什么意义 TODO delete
            input_query_bbox: (B, pad_size, 7)
            attn_mask: (1000+pad_size, 1000+pad_size)
            dn_meta: {'pad_size': 2 * max_gt_num * 3, 'num_dn_group': 3}
            其中2 * max_gt_num * 3 = pad_size, n1 n2是batchsize=2时的假设, n2>n1, 这个乘法式表示有三组噪声，每组分正负样本，正样本做重建，负样本要学会剔除
            '''
            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(targets, dn['dn_number'],dn['dn_label_noise_ratio'], dn['dn_box_noise_scale']),
                training=self.train_flag,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_channel,
                label_enc=None,
                code_size=self.code_size,
            ) # 这一步准备噪声样本
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            targets = None

        outputs = self.transformer(
            features, # [(B, 256, H, W)]
            pos_encodings, # [(B, 256, H, W)]
            input_query_bbox, # (B, pad_size, 7)
            input_query_label, # (B, pad_size, num_classes) num_classes=1
            attn_mask, # (1000+pad_size, 1000+pad_size)
            targets=targets, # [Sample1:Dict, Sample2:Dict...]
        )
        '''
        hidden_state: (6, B, pad_size + 1000 + 4*max_gt_num, 256) pad_size其实等于 6*max_gt_num 这是6层decoder的输出
        init_reference: (B, pad_size + 1000 + 4*max_gt_num, 7) 6批噪声gt+初始的dqs得到的1000个box 加上 4批 gt, 第一批是gt,后面3批示噪声gt正样本
        inter_references: (6, B, pad_size + 1000 + 4*max_gt_num, 10) pad_size其实等于 6*max_gt_num。 这是每一层的预测结果
        src_embed: (B, H*W, 256) 粗查询, 经过了一层DGA layer后scatter回去
        src_ref_windows: (B, H * W, 7) 参考框，类似于锚框
        src_indexes: (B, 1000, 1) 1000个fined dqs query 索引
        '''
        hidden_state, init_reference, inter_references, src_embed, src_ref_windows, src_indexes = outputs

        # decoder
        outputs_classes = []
        outputs_coords = []
        outputs_ious = []

        for idx in range(hidden_state.shape[0]): # 这里是遍历6层
            if idx == 0:
                reference = init_reference
            else:
                reference = inter_references[idx - 1]
            outputs_class, outputs_coord, outputs_iou = self.transformer.decoder.detection_head(hidden_state[idx], # 每一层明明已经产出过对应的结果了，为什么有又进行一遍输出头？答：对比学习的辅助输出参数不同步，这里应该是这个考虑？
                                                                                                reference, idx) # 根据每一层的query特征，和reference重新来一次输出头，但是与之前不一样的是，这次会带上GT以及GT噪声正样本， 这部分来自于对比学习
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_ious.append(outputs_iou)

        outputs_class = torch.stack(outputs_classes) # (6, B, pad_size + 1000 + 4*max_gt_num, 1)
        outputs_coord = torch.stack(outputs_coords) # (6, B, pad_size + 1000 + 4*max_gt_num, 7)
        outputs_iou = torch.stack(outputs_ious) # (6, B, pad_size + 1000 + 4*max_gt_num, 1)

        # dn post process
        '''
        去噪组分开
        outputs_class:  (6, B, 1000 + 4*max_gt_num, 1)
        outputs_coord:  (6, B, 1000 + 4*max_gt_num, 7)
        outputs_iou:    (6, B, 1000 + 4*max_gt_num, 1)

        dn_meta = {
            "pad_size":                 2 * max_gt_num * 3
            "num_dn_group":             3
            "output_known_lbs_bboxes":  {
                "pred_logits":  (B, pad_size, 1) 最后一层的预测结果 这个是噪声样本
                "pred_boxes":   (B, pad_size, 7) 
                "pred_ious":    (B, pad_size, 1) 

                "aux_outputs":  List[Dict{"pred_logits": (B, pad_size, 3), "pred_boxes": (B, pad_size, 7), "pred_ious": (B, pad_size, 1)}...] 五个，表示每一层的噪声样本                 
            }
        }
        '''
        if dn['dn_number'] > 0 and dn_meta is not None:
            outputs_class, outputs_coord, outputs_iou = dn_post_process_w_ious(
                outputs_class,
                outputs_coord,
                outputs_iou,
                dn_meta,
                self.aux_loss, # 使用辅助损失
                self._set_aux_loss,
            )

        # only for supervision
        enc_outputs = None
        if self.train_flag: # 防止梯度流被污染，DQS只是辅助筛选query，筛选时不能参与梯度计算，否则训练早期低质量的query会大幅度影响结果，因此在DQS中必须要detach
            enc_class, enc_coords, enc_ious = self.transformer.proposal_head(src_embed, src_ref_windows) # 这个之前是dqs打分用的，这里输入的是和当时一样的输入，即获得当时的打分结果，注意，筛选操作存在detach操作，分开到这里做损失实际上是防止梯度流被污染
            enc_outputs = {
                'topk_indexes': src_indexes,    # (B, 1000, 1) # 通过dqs挑选的1000个query的索引
                'pred_logits': enc_class,       # (B, H*W, 1)
                'pred_boxes': enc_coords,       # (B, H*W, 7)
                'pred_ious': enc_ious,      # (B, H*W, 1)

            }

        # compute decoder losses
        outputs = {
            # "pred_scores_mask": score_mask, # (B, H, W)
            "pred_logits": outputs_class[-1][:, : self.num_queries],    # (B, 1000, 1) # 最后一层
            "pred_boxes": outputs_coord[-1][:, : self.num_queries],     # (B, 1000, 7)
            'pred_ious': outputs_iou[-1][:, : self.num_queries],        # (B, 1000, 1)

            "aux_outputs": self._set_aux_loss(
                outputs_class[:-1, :, : self.num_queries], outputs_coord[:-1, :, : self.num_queries],
                outputs_iou[:-1, :, : self.num_queries],  # List[Dict{"pred_logits": (B, 1000, 3), "pred_boxes": (B, 1000, 7), ...] 5个元素 表示前五层的1000个query
            ),
        }
        if self.train_flag:
            '''
            pred_dicts:         {"enc_outputs":  Dict()
                                "outputs":      Dict()}
            outputs_class:      (6, B, 1000 + 4*max_gt_num, 1)
            outputs_coord:      (6, B, 1000 + 4*max_gt_num, 7)
            dn_meta:            Dict() 去噪数据
            '''
            pred_dicts = dict(enc_outputs=enc_outputs, outputs=outputs)
            return pred_dicts, outputs_class, outputs_coord, dn_meta
        else:
            pred_dicts = dict(enc_outputs=enc_outputs, outputs=outputs)
            return pred_dicts

    def forward(self, batch_dict):
        if self.train_flag:
            pred_dicts, outputs_class, outputs_coord, dn_meta = self.predict(batch_dict)
        else:
            pred_dicts = self.predict(batch_dict)

        if not self.train_flag:
            bboxes = self.get_bboxes(pred_dicts)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_bboxes_3d = batch_dict['object_bbx_center'] # (B, maxnum, 7)
            gt_bboxes_3d_mask = batch_dict['object_bbx_mask'] # (B, maxnum)
            gt_labels_3d = gt_bboxes_3d.new_ones(gt_bboxes_3d_mask.size(0), gt_bboxes_3d_mask.size(1))
            gt_labels_3d = gt_labels_3d.long() - 1 # (B, maxnum)

            loss, tb_dict = self.loss(gt_bboxes_3d, gt_bboxes_3d_mask, gt_labels_3d, pred_dicts, dn_meta, outputs_class, outputs_coord)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_bboxes(self, pred_dicts):
        outputs = pred_dicts['outputs']
        out_logits = outputs['pred_logits'] # (B, 1000, 1) B在验证或者测试的时候一定是 ==1
        out_bbox = outputs['pred_boxes'] # (B, 1000, 7)
        out_iou = (outputs['pred_ious'] + 1) / 2 # (B, 1000, 1) 映射到0-1
        batch_size = out_logits.shape[0]

        out_iou = out_iou.repeat([1, 1, out_logits.shape[-1]])
        out_iou = out_iou.view(out_logits.shape[0], -1) # （B, 1000）

        out_prob = out_logits.sigmoid()
        out_prob = out_prob.view(out_logits.shape[0], -1) # (B, 1000)
        out_bbox = self.decode_bbox(out_bbox)

        def _process_output(indices, bboxes):
            topk_boxes = indices.div(out_logits.shape[2], rounding_mode="floor").unsqueeze(-1)
            labels = indices % out_logits.shape[2] # 得到标签
            boxes = torch.gather(bboxes, 0, topk_boxes.repeat(1, out_bbox.shape[-1]))
            return labels + 1, boxes, topk_boxes

        new_ret_dict = []
        for i in range(batch_size):
            out_prob_i = out_prob[i] # （1000，）
            out_bbox_i = out_bbox[i] # (1000, 7)

            out_iou_i = out_iou[i] # (1000, )

            # 用 qs 来输出
            # mask = torch.ones_like(out_prob_i) # (1000, ) 全1
            # score_mask = out_prob_i > 0.2 # (1000, )  大于阈值的全部置为True
            # mask = mask * score_mask.type_as(mask) # 再将Cyclist对应的部分mask掉，这是因为这种类别不均衡切置信度一般不高，可能引入噪声会影响IoU校准

            # if isinstance(self.iou_rectifier, float):
            #     temp_probs = torch.pow(out_prob_i, 1 - self.iou_rectifier) * torch.pow(out_iou_i, self.iou_rectifier)
            #     out_prob_i = out_prob_i * (1 - mask) + mask * temp_probs
            # else:
            #     raise TypeError('only list or float')
            
            '''
            # out_prob_i_ori = out_prob_i.view(out_bbox_i.shape[0], -1)  # [1000, 3]
            # max_out_prob_i, pred_cls = torch.max(out_prob_i_ori, dim=-1)
            # out_iou_i_ori = out_iou_i.view(out_bbox_i.shape[0], -1)
            #
            # out_prob_i_list = []
            # out_bbox_i_list = []
            # out_iou_i_list = []
            #
            # ONLY_FOR_CAR = True
            #
            # for cls in range(self.num_classes):
            #     cls_mask = pred_cls == cls
            #     if cls_mask.sum() >= 1:
            #         out_prob_i_cls_valid = out_prob_i_ori[cls_mask]
            #         out_bbox_i_cls_valid = out_bbox_i[cls_mask]
            #         out_iou_i_cls_valid = out_iou_i_ori[cls_mask]
            #         max_out_prob_i_cls_valid = max_out_prob_i[cls_mask]
            #
            #         if ONLY_FOR_CAR & (cls == 0):  # 0 is for CAR
            #             out_bbox_i_xy = (out_bbox_i_cls_valid[:, :2] - torch.Tensor(
            #                 self.point_cloud_range[0:2]).unsqueeze(0).type_as(
            #                 out_bbox_i_cls_valid)) * 4  # 4 for large objects, such as car
            #             out_bbox_i_xy = torch.round(out_bbox_i_xy).int()
            #             sort_xy = out_bbox_i_xy[:, 0] * 1000 + out_bbox_i_xy[:, 1]
            #             unq_coords, unq_inv, unq_cnt = torch.unique(sort_xy, return_inverse=True, return_counts=True,
            #                                                         dim=0)
            #
            #             out, valid_mask = scatter_max(max_out_prob_i_cls_valid, unq_inv)
            #
            #             out_prob_i_tmp = out_prob_i_cls_valid[valid_mask].view(-1)
            #             out_bbox_i_tmp = out_bbox_i_cls_valid[valid_mask]
            #             out_iou_i_tmp = out_iou_i_cls_valid[valid_mask].view(-1)
            #
            #         else:
            #             out_bbox_i_xy = (out_bbox_i_cls_valid[:, :2] - torch.Tensor(
            #                 self.point_cloud_range[0:2]).unsqueeze(0).type_as(
            #                 out_bbox_i_cls_valid)) * 20
            #             out_bbox_i_xy = torch.round(out_bbox_i_xy).int()
            #             sort_xy = out_bbox_i_xy[:, 0] * 1000 + out_bbox_i_xy[:, 1]
            #             unq_coords, unq_inv, unq_cnt = torch.unique(sort_xy, return_inverse=True, return_counts=True,
            #                                                         dim=0)
            #
            #             out, valid_mask = scatter_max(max_out_prob_i_cls_valid, unq_inv)
            #
            #             out_prob_i_tmp = out_prob_i_cls_valid[valid_mask].view(-1)
            #             out_bbox_i_tmp = out_bbox_i_cls_valid[valid_mask]
            #             out_iou_i_tmp = out_iou_i_cls_valid[valid_mask].view(-1)
            #
            #         out_prob_i_list.append(out_prob_i_tmp)
            #         out_bbox_i_list.append(out_bbox_i_tmp)
            #         out_iou_i_list.append(out_iou_i_tmp)
            #
            # out_prob_i = torch.cat(out_prob_i_list, dim=0)
            # out_bbox_i = torch.cat(out_bbox_i_list, dim=0)
            # out_iou_i = torch.cat(out_iou_i_list, dim=0)
            '''
            topk_indices_i = torch.nonzero(out_prob_i >= 0.1, as_tuple=True)[0] # 筛选置信度大于0.1的的索引 (n, )
            scores = out_prob_i[topk_indices_i] # (n, ) 这个因为多cls也是相同的repeat 所以不用上面的操作

            labels, boxes, topk_indices = _process_output(topk_indices_i.view(-1), out_bbox_i) # 分别得到标签和bbox shape 为 (n, ) and (n, 7)

            ious = out_iou_i[topk_indices_i] # (n, )

            scores_list = list()
            labels_list = list()
            boxes_list = list() 

            for c in range(self.num_classes):
                mask = (labels - 1) == c # 对于分类无关来说其实是全True ，(n, ), 对于多分类的来说其实就是依次处理每个分类用的
                scores_temp = scores[mask]
                ious_temp = ious[mask]
                labels_temp = labels[mask]
                boxes_temp = boxes[mask]

                if isinstance(self.iou_rectifier, list):
                    iou_rectifier = torch.tensor(self.iou_rectifier).to(out_prob)[c]
                    scores_temp = torch.pow(scores_temp, 1 - iou_rectifier) * torch.pow(ious_temp,
                                                                                        iou_rectifier)
                elif isinstance(self.iou_rectifier, float): # 类别无关 0.68 这又是在算质量得分
                    scores_temp = torch.pow(scores_temp, 1 - self.iou_rectifier) * torch.pow(ious_temp,
                                                                                                self.iou_rectifier)
                else:
                    raise TypeError('only list or float')

                scores_list.append(scores_temp)
                labels_list.append(labels_temp)
                boxes_list.append(boxes_temp)

            scores = torch.cat(scores_list, dim=0) # (n,)
            labels = torch.cat(labels_list, dim=0) # (n,) 在类别无关中，其实label是全0
            boxes = torch.cat(boxes_list, dim=0) # (n,7)
            ret = dict(pred_boxes=boxes, pred_scores=scores, pred_labels=labels)
            new_ret_dict.append(ret)

        return new_ret_dict

    def compute_losses(self, outputs, targets, dn_meta=None, supervise_iou=False):
        loss_dict = self.losses(outputs, targets, dn_meta=dn_meta, supervise_iou=supervise_iou)

        weight_dict = self.losses.weight_dict
        for k, v in loss_dict.items():
            if k in weight_dict:
                loss_dict[k] = v * weight_dict[k]

        return loss_dict

    def compute_score_losses(self, pred_scores_mask, gt_bboxes_3d, gt_bboxes_3d_mask, foreground_mask):
        '''
        pred_scores_mask: 前景预测, (B, H, W)
        gt_bboxes_3d: (B, max_num, 7)
        '''
        gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
        grid_size = torch.ceil(torch.from_numpy(self.grid_size).to(gt_bboxes_3d) / self.feature_map_stride) # 空间八倍下采样后的尺寸 [252, 100, 50/8]
        pc_range = torch.from_numpy(np.array(self.point_cloud_range)).to(gt_bboxes_3d) # [-100.8, -40, -3.5, 100.8, 40, 1.5] 点云尺寸
        stride = (pc_range[3:5] - pc_range[0:2]) / grid_size[0:2] # 实际尺寸和特征图的差距
        gt_score_map = list()
        yy, xx = torch.meshgrid(torch.arange(grid_size[1]), torch.arange(grid_size[0])) # 两个都是（100， 252）
        points = torch.stack([yy, xx]).permute(1, 2, 0).flip(-1) # (100, 252, 2) 最后一个反转操作将存储方法设置为(x,y)格式
        points = torch.cat([points, torch.ones_like(points[..., 0:1]) * 0.5], dim=-1).reshape([-1, 3]) # （100*252， 3）新增的维度里面存的都是0.5 也就是z轴坐标都是0.5
        for i in range(len(gt_bboxes_3d)):
            boxes = gt_bboxes_3d[i] # （max_num, 7）
            boxes_mask = gt_bboxes_3d_mask[i].bool() # (max_num,)
            boxes = boxes[boxes_mask] # (n_i, 7)
            # boxes = boxes[(boxes[:, 3] > 0) & (boxes[:, 4] > 0)] # 这筛选出有效的部分
            ones = torch.ones_like(boxes[:, 0:1]) # (n_i, 1)
            bev_boxes = torch.cat([boxes[:, 0:2], ones * 0.5, boxes[:, 3:5], ones * 0.5, boxes[:, 6:7]], dim=-1) # 去除z轴，全部填充0.5 (n_i, 7)
            bev_boxes[:, 0:2] -= pc_range[0:2] # 减去边界最小值 得到相对偏移
            bev_boxes[:, 0:2] /= stride # 得到在特征图中的位置
            bev_boxes[:, 3:5] /= stride # 得到在特征图中的长宽

            box_ids = roiaware_pool3d_utils.points_in_boxes_gpu(
                points[:, 0:3].unsqueeze(dim=0).float().cuda(), # （1， HW， 3）
                bev_boxes[:, 0:7].unsqueeze(dim=0).float().cuda() # （1, n_i, 7）
            ).long().squeeze(dim=0).cpu().numpy() # (1, HW) 不等于-1的部分就是没有落在任何一个box中
            box_ids = box_ids.reshape([grid_size[1].long(), grid_size[0].long()]) # (100, 252) 
            mask = torch.from_numpy(box_ids != -1).to(bev_boxes) # (100, 252) 
            gt_score_map.append(mask)
        gt_score_map = torch.stack(gt_score_map) # (B, 100, 252) 在box的部分被标记为True

        num_pos = max(gt_score_map.eq(1).float().sum().item(), 1) # max保证至少为1，这是算所有位置的个数，也就是所有前景点的个数
        
        loss_score = ClassificationLoss.sigmoid_focal_loss(
            pred_scores_mask.flatten(0), # 展平 (BHW, )
            gt_score_map.flatten(0), # (BHW)
            0.25,
            gamma=2.0,
            reduction="sum",
        )
        loss_score /= num_pos

        return loss_score

    def loss(self, gt_bboxes_3d, gt_bboxes_3d_mask, gt_labels_3d, pred_dicts, dn_meta=None, outputs_class=None, outputs_coord=None):
        loss_all = 0
        loss_dict = dict()
        targets = list()

        for batch_idx in range(len(gt_bboxes_3d)): # 遍历每个样本
            target = {}
            gt_bboxes = gt_bboxes_3d[batch_idx] # (max_num, 7)
            gt_bboxes_mask = gt_bboxes_3d_mask[batch_idx] # (max_num, )
            gt_labels = gt_labels_3d[batch_idx] # (max_num, )

            valid_bboxes = gt_bboxes[gt_bboxes_mask.bool()]
            valid_labels = gt_labels[gt_bboxes_mask.bool()]

            target['gt_boxes'] = self.encode_bbox(valid_bboxes) # boxes本身是torch.float64 这个encode会让它变成torch.float32
            target['labels'] = valid_labels
            targets.append(target)

        enc_outputs = pred_dicts['enc_outputs']
        bin_targets = copy.deepcopy(targets)
        # [tgt["labels"].fill_(0) for tgt in bin_targets] NOTE 这个是Github Issue提出的，注释掉后带来了0.2-0.3的提升，这是为什么？ 如果不注释，其实就变成类别无关预测 答：其实这里就是ConQuer的代码中的设置 作者忘删除了，seed 中的dqs就是需要多分类的
        dqs_losses = self.compute_losses(enc_outputs, bin_targets) # dqs的结果先作为detect输出监督dqs质量选择
        for k, v in dqs_losses.items():
            loss_all += v
            loss_dict.update({k + "_enc": v.item()})
        # for k, v in dqs_losses.items():
        #     loss_dict.update({k + "_debug": v})
        outputs = pred_dicts['outputs']
        dec_losses = self.compute_losses(outputs, targets, dn_meta, supervise_iou=True)
        for k, v in dec_losses.items():
            loss_all += v
            loss_dict.update({k: v.item()})  # 这里包含了最后一层的检测结果损失，还有dn的去噪损失，以及辅助输出的五层的相应的检测和去噪损失

        # compute contrastive loss
        if dn_meta is not None:
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [n1, n2, ...]
            max_gt = max(per_gt_num)
            num_gts = sum(per_gt_num)
            if num_gts > 0:
                for li in range(self.model_cfg["num_decoder_layers"]): # 6层decoder
                    contrastive_loss = 0.0
                    projs = torch.cat((outputs_class[li], outputs_coord[li]), dim=-1) # (B, 1000 + 4*max_gt_num, 1+7) 某一层的输出
                    gt_projs = self.projector(projs[:, self.num_queries:].detach()) # gt 线性变化 # (B, 4*max_gt_num, 256) 注意这四批的gt前一批是真gt，后面三个则是噪声正样本 注意，这里detach是因为这里不能梯度回传，这个是将其映射到高维空间
                    pred_projs = self.predictor(self.projector(projs[:, : self.num_queries])) # (B, 1000, 256) query需要额外的MLP，出自ConQueR论文中的设计
                    # num_gts x num_locs

                    pos_idxs = list(range(1, dn_meta["num_dn_group"] + 1)) # [1, 2, 3]
                    for bi, idx in enumerate(outputs["matched_indices"]): # 这个是[((n1,), (n1,)), ...]匹配结果
                        sim_matrix = (
                                self.similarity_f(
                                    gt_projs[bi].unsqueeze(1),
                                    pred_projs[bi].unsqueeze(0),
                                )
                                / self.tau
                        )# 求得相似度矩阵(4*max_gt_num, 1000)
                        matched_pairs = torch.stack(idx, dim=-1) # (n1, 2) 以第一个样本为例 
                        neg_mask = projs.new_ones(self.num_queries).bool() # （1000，）
                        neg_mask[matched_pairs[:, 0]] = False # 最佳匹配的query标记成False, 换言之，没匹配上的都是True HACK 这里有个问题，负样本不应该是999个吗，它这样相当于负样本变少了，即使匹配上了，彼此之间应该还是负样本
                        for pair in matched_pairs: # 遍历配对后的每一对
                            pos_mask = torch.tensor([int(pair[1] + max_gt * pi) for pi in pos_idxs],
                                                    device=sim_matrix.device) # 这个是用来筛选gt的 明明是4*max_gt_num，但是是选择了后面三批带噪声的gt，也就是选了三个(3, )
                            pos_pair = sim_matrix[pos_mask, pair[0]].view(-1, 1) # 正样本 （3，1）
                            neg_pairs = sim_matrix[:, neg_mask][pos_mask] # 负样本，（3，1000-n1） XXX 注意这里是不是有问题，这是为了简化逻辑？❓
                            loss_gti = (
                                    torch.log(torch.exp(pos_pair) + torch.exp(neg_pairs).sum(dim=-1, keepdim=True))
                                    - pos_pair
                            ) # （3， 1） 
                            contrastive_loss += loss_gti.mean() # 3组gt对比，求均值再加上去
                    loss_contrastive_dec_li = self.contras_loss_coeff * contrastive_loss / num_gts # 乘上系数=0.2后要除以gt总数，以均衡不同样本的gt数不同引起的数值波动
                    loss_all += loss_contrastive_dec_li
                    loss_dict.update({'loss_contrastive_dec_' + str(li): loss_contrastive_dec_li.item()})

        # pred_scores_mask = outputs['pred_scores_mask'] # (B, H, W)
        # loss_score = self.compute_score_losses(pred_scores_mask, gt_bboxes_3d.to(torch.float32), gt_bboxes_3d_mask, None) # 前景预测损失
        # loss_all += loss_score
        # loss_dict.update({'loss_score': loss_score.item()})

        return loss_all, loss_dict

    def encode_bbox(self, bboxes): # 输入的是n, 7
        z_normalizer = 10
        targets = torch.zeros([bboxes.shape[0], self.code_size]).to(bboxes.device) # n, 7 同时这里有一个隐式数据类型转变将target 类型从torch.float64变成torch.float32 如果没有这个过程，会报各种类型错误
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

    def decode_bbox(self, pred_boxes):
        z_normalizer = 10

        pred_boxes[..., 0] = pred_boxes[..., 0] * (self.point_cloud_range[3] - self.point_cloud_range[0]) + \
                             self.point_cloud_range[0]
        pred_boxes[..., 1] = pred_boxes[..., 1] * (self.point_cloud_range[4] - self.point_cloud_range[1]) + \
                             self.point_cloud_range[1]
        pred_boxes[..., 2] = pred_boxes[..., 2] * 2 * z_normalizer + -1 * z_normalizer
        pred_boxes[..., 3] = pred_boxes[..., 3] * (self.point_cloud_range[3] - self.point_cloud_range[0])
        pred_boxes[..., 4] = pred_boxes[..., 4] * (self.point_cloud_range[4] - self.point_cloud_range[1])
        pred_boxes[..., 5] = pred_boxes[..., 5] * 2 * z_normalizer
        pred_boxes[..., -1] = pred_boxes[..., -1] * np.pi * 2 - np.pi
        if self.code_size > 7:
            pred_boxes[:, 7] = (pred_boxes[:, 7]) * (self.point_cloud_range[3] - self.point_cloud_range[0])
            pred_boxes[:, 8] = (pred_boxes[:, 8]) * (self.point_cloud_range[4] - self.point_cloud_range[1])
        return pred_boxes

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou):
        return [{"pred_logits": a, "pred_boxes": b, "pred_ious": c} for a, b, c in
                zip(outputs_class, outputs_coord, outputs_iou)]

# vfl loss
class VariFocalLoss(nn.Module):
    def __init__(self, focal_alpha, decode_func=None):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.target_classes = None
        self.src_logits = None
        self.alpha = 0.25
        self.gamma = 2.0
        self.decode_func = decode_func

    def forward(self, outputs, targets, indices, num_boxes):
        outputs["matched_indices"] = indices
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] # dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 
        target_classes_onehot = torch.zeros_like(src_logits) # (B, H*W, 1) det损失时为 (B, 1000, 1) 
        
        idx = _get_src_permutation_idx(indices) # 返回两个索引量，[batch索引(n_all, )，最佳匹配query索引(n_all, )]

        target_boxes = torch.cat([t['gt_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # (n_all, 7)
        target_boxes = self.decode_func(target_boxes)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 索引到最佳匹配的gt label (n_all, ) n_all=n1+n2+...

        target_score_o = torch.zeros(target_classes_onehot.size(0), target_classes_onehot.size(1), dtype=src_logits.dtype, device=src_logits.device)

        # for metrics calculation
        self.target_classes = target_classes_o

        if "topk_indexes" in outputs.keys(): # 这是dqs做损失要的，dqs也和最佳匹配的做损失，要求最佳匹配的query的置信度接近1
            topk_indexes = outputs["topk_indexes"] # (B, 1000, 1) dqs结果索引
            self.src_logits = torch.gather(src_logits, 1, topk_indexes.expand(-1, -1, src_logits.shape[-1]))[idx] # (B, 1000, 1)
            target_classes_onehot[idx[0], topk_indexes[idx].squeeze(-1), target_classes_o] = 1 # 索引到对应的位置 根据不同的类别，索引后为1，形成了one-hot编码

            src_boxes = outputs['pred_boxes'][idx[0], topk_indexes[idx].squeeze(-1)] # (n_all. 7)
            src_boxes = self.decode_func(src_boxes)
            ious = iou3d_nms_utils.boxes_iou3d_gpu(src_boxes, target_boxes) # (n_all, n_all)
            ious = torch.diag(ious).detach() # 返回的是一个1D张量 (n_all,) 预测与其gt对应的iou
            target_score_o[idx[0], topk_indexes[idx].squeeze(-1)] = ious.to(target_score_o.dtype) # (B, HW)
            # print('sum target is ', target_classes_onehot.sum())
        else:
            self.src_logits = src_logits[idx]
            # 0 for bg, 1 for fg
            # N, L, C
            target_classes_onehot[idx[0], idx[1], target_classes_o] = 1
            src_boxes = outputs['pred_boxes'][idx[0], idx[1]] # (n_all. 7)
            src_boxes = self.decode_func(src_boxes)
            ious = iou3d_nms_utils.boxes_iou3d_gpu(src_boxes, target_boxes) # (n_all, n_all)
            ious = torch.diag(ious).detach() # 返回的是一个1D张量 (n_all,) 预测与其gt对应的iou
            target_score_o[idx] = ious.to(target_score_o.dtype) # (B, HW)

        
        target_score = target_score_o.unsqueeze(-1) * target_classes_onehot # （B, HW, 1） det损失时为 (B, 1000, 1)  标签对应的那个类变成一个分数

        pred_score = F.sigmoid(src_logits).detach() # （B, HW, 1） det损失时为 (B, 1000, 1)
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target_classes_onehot) + target_score

        loss_ce = (
                 F.binary_cross_entropy_with_logits(
                    src_logits, # 预测，dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 
                    target_score, # dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 匹配上的query
                    weight=weight,
                    reduction="sum"
                )
                / num_boxes
        ) # focal loss 取mean结果

        losses = {
            "loss_ce": loss_ce,
        }

        return losses

# mal loss
class MatchabilityAwareLoss(nn.Module):
    def __init__(self, focal_alpha, decode_func=None):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.target_classes = None
        self.src_logits = None
        self.mal_alpha = None
        self.gamma = 1
        self.decode_func = decode_func

    def forward(self, outputs, targets, indices, num_boxes):
        outputs["matched_indices"] = indices
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] # dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 
        target_classes_onehot = torch.zeros_like(src_logits) # (B, H*W, 1) det损失时为 (B, 1000, 1) 
        
        idx = _get_src_permutation_idx(indices) # 返回两个索引量，[batch索引(n_all, )，最佳匹配query索引(n_all, )]

        target_boxes = torch.cat([t['gt_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # (n_all, 7)
        target_boxes = self.decode_func(target_boxes)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 索引到最佳匹配的gt label (n_all, ) n_all=n1+n2+...
        # print("indices is ", indices)
        # print("t['labels'] is ", targets[0]['labels'])
        # print("target_classes_o is ", target_classes_o)

        # for metrics calculation
        self.target_classes = target_classes_o

        target_score_o = torch.zeros(target_classes_onehot.size(0), target_classes_onehot.size(1), dtype=src_logits.dtype, device=src_logits.device) # (B, H*W) det损失时为 (B, 1000) 

        if "topk_indexes" in outputs.keys(): # 这是dqs做损失要的，dqs也和最佳匹配的做损失，要求最佳匹配的query的置信度接近1
            topk_indexes = outputs["topk_indexes"] # (B, 1000, 1) dqs结果索引
            self.src_logits = torch.gather(src_logits, 1, topk_indexes.expand(-1, -1, src_logits.shape[-1]))[idx] # (B, 1000, 1)
            target_classes_onehot[idx[0], topk_indexes[idx].squeeze(-1), target_classes_o] = 1 # 索引到对应的位置 根据不同的类别，索引后为1，形成了one-hot编码

            src_boxes = outputs['pred_boxes'][idx[0], topk_indexes[idx].squeeze(-1)] # (n_all. 7)
            src_boxes = self.decode_func(src_boxes)
            ious = iou3d_nms_utils.boxes_iou_bev(src_boxes, target_boxes) # (n_all, n_all)
            # print("outputs['pred_boxes'] shape ", outputs['pred_boxes'].shape)
            # print("ious shape ", ious.shape)

            ious = torch.diag(ious).detach() # 返回的是一个1D张量 (n_all,) 预测与其gt对应的iou
            target_score_o[idx[0], topk_indexes[idx].squeeze(-1)] = ious.to(target_score_o.dtype) # (B, HW)
            # print('sum target is ', target_classes_onehot.sum())
            # print('ious shape is ', ious.shape)
        else:
            self.src_logits = src_logits[idx]
            # 0 for bg, 1 for fg
            # N, L, C
            target_classes_onehot[idx[0], idx[1], target_classes_o] = 1
            src_boxes = outputs['pred_boxes'][idx[0], idx[1]] # (n_all. 7)
            src_boxes = self.decode_func(src_boxes)
            ious = iou3d_nms_utils.boxes_iou_bev(src_boxes, target_boxes) # (n_all, n_all)
            ious = torch.diag(ious).detach() # 返回的是一个1D张量 (n_all,) 预测与其gt对应的iou
            target_score_o[idx] = ious.to(target_score_o.dtype) # (B, HW)

        # print("------------------------------------------------------------------------------------")
        # print("target_score_o shape ", target_score_o.shape)
        # print("target_classes_onehot shape ", target_classes_onehot.shape)
        # print("ious shape ", ious.shape)
        # print("ious ==> ", ious)

        target_score = target_score_o.unsqueeze(-1) * target_classes_onehot # （B, HW, 1） det损失时为 (B, 1000, 1)  标签对应的那个类变成一个分数

        pred_score = F.sigmoid(src_logits).detach() # （B, HW, 1） det损失时为 (B, 1000, 1)
        target_score = target_score.pow(self.gamma) # NOTE paper上的公式应该是有笔误 
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target_classes_onehot) + target_classes_onehot
        else:
            weight = pred_score.pow(self.gamma) * (1 - target_classes_onehot) + target_classes_onehot

        loss_ce = (
                 F.binary_cross_entropy_with_logits(
                    src_logits, # 预测，dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 
                    target_score, # dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 匹配上的query
                    weight=weight,
                    reduction="sum"
                )
                / num_boxes
        ) # mal loss 取mean结果

        losses = {
            "loss_ce": loss_ce,
        }
        # print("weight ==> ", weight)
        # print("loss_ce ==> ", loss_ce)
        # xxx
        # print("------------------------------------------------------------------------------------")

        return losses

class ClassificationLoss(nn.Module):
    def __init__(self, focal_alpha):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.target_classes = None
        self.src_logits = None

    @staticmethod
    def sigmoid_focal_loss(
            logits,
            targets,
            alpha: float = -1,
            gamma: float = 2,
            reduction: str = "none",
    ):
        # 输入的两项形状都为 (B, H*W, 1) 或者 det损失时是(B, 1000, 1) 预测结果 & one-hot编码
        p = torch.sigmoid(logits)

        ce_loss = F.binary_cross_entropy(p, targets, reduction="none") # 二元交叉熵 (B, H*W, 1) 或者 det损失时是(B, 1000, 1) 
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma) # 包含调节因子，初步形成focal loss

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # 正样本则调节因子为alpha，负样本则为1-alpha
            loss = alpha_t * loss # focal loss 完成， (B, H*W, 1)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum() # 求和 常量

        return loss

    def forward(self, outputs, targets, indices, num_boxes):
        outputs["matched_indices"] = indices
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] # dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 
        target_classes_onehot = torch.zeros_like(src_logits) # (B, H*W, 1) det损失时为 (B, 1000, 1) 

        idx = _get_src_permutation_idx(indices) # 返回两个索引量，[batch索引(n_all, )，最佳匹配query索引(n_all, )]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 索引到最佳匹配的gt label (n_all, )

        # for metrics calculation
        self.target_classes = target_classes_o

        if "topk_indexes" in outputs.keys(): # 这是dqs做损失要的，dqs也和最佳匹配的做损失，要求最佳匹配的query的置信度接近1
            topk_indexes = outputs["topk_indexes"] # (B, 1000, 1) dqs结果索引
            self.src_logits = torch.gather(src_logits, 1, topk_indexes.expand(-1, -1, src_logits.shape[-1]))[idx] # (B, 1000, 1)
            target_classes_onehot[idx[0], topk_indexes[idx].squeeze(-1), target_classes_o] = 1 # 索引到对应的位置 根据不同的类别，索引后为1，形成了one-hot编码
        else:
            self.src_logits = src_logits[idx]
            # 0 for bg, 1 for fg
            # N, L, C
            target_classes_onehot[idx[0], idx[1], target_classes_o] = 1

        loss_ce = (
                self.sigmoid_focal_loss(
                    src_logits, # 预测，dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 
                    target_classes_onehot, # dqs损失时是 (B, H*W, 1) det损失时为 (B, 1000, 1) 匹配上的query
                    alpha=self.focal_alpha, # 0.25
                    gamma=2.0,
                    reduction="sum"
                )
                / num_boxes
        ) # focal loss 取mean结果

        losses = {
            "loss_ce": loss_ce,
        }

        return losses


class RegressionLoss(nn.Module):
    def __init__(self, decode_func=None):
        super().__init__()
        self.decode_func = decode_func

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        ''''
        outputs: Dict, DQS结果或者是模型输出的结果
        targets: List [Dict{}, Dict{}...] batch中按样本分结果
        indices: List [((n,), (n,)), ...] 最小成本匹配索引 或者是 去噪gt样本索引
        num_boxes: batch中所有gt数总和
        '''
        idx = _get_src_permutation_idx(indices) # 返回两个索引量，[batch索引(n_all, )，最佳匹配query索引(n_all, )] 其中第二个也可能是去噪正样本（3*(n1-1) +  3*(n2-1), ....), ）

        if "topk_indexes" in outputs.keys():
            pred_boxes = torch.gather(
                outputs["pred_boxes"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_boxes"].shape[-1]),
            ) # （B, 1000, 7）
            pred_ious = torch.gather(
                outputs["pred_ious"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_ious"].shape[-1]),
            ) # （B, 1000, 1）
        else:
            pred_boxes = outputs["pred_boxes"] # （B, 1000, 7）如果是去噪gt （B, pad_size, 7）
            pred_ious = outputs["pred_ious"] # （B, 1000, 1）

        target_boxes = torch.cat([t["gt_boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0) # 索引到最佳匹配的gt (n1+n2+..., 7) 也可能是去噪正样本（3*(n1-1) +  3*(n2-1), ....), ）

        src_boxes, src_rads = pred_boxes[idx].split(6, dim=-1) # (n_all, 6), (n_all, 1)
        target_boxes, target_rads = target_boxes.split(6, dim=-1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_rad = F.l1_loss(src_rads, target_rads, reduction="none")

        gt_iou = torch.diag(
            generalized_box3d_iou(
                box_cxcyczlwh_to_xyxyxy(src_boxes), # 最佳匹配的query
                box_cxcyczlwh_to_xyxyxy(target_boxes), # 最佳匹配的gt
            )
        ) # 返回一个一维张量 (n_all, )即最佳匹配之间的giou

        loss_giou = 1 - gt_iou

        losses = {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
            "loss_rad": loss_rad.sum() / num_boxes,
        }

        pred_ious = pred_ious[idx] # （n_all, 1）

        box_preds = self.decode_func(torch.cat([src_boxes, src_rads], dim=-1)) # 反归一化
        box_target = self.decode_func(torch.cat([target_boxes, target_rads], dim=-1))
        iou_target = iou3d_nms_utils.paired_boxes_iou3d_gpu(box_preds, box_target) # (n_all, ) iou
        iou_target = iou_target * 2 - 1 # (0, 1) map 到 (-1, 1)
        iou_target = iou_target.detach()
        loss_iou = F.l1_loss(pred_ious, iou_target.unsqueeze(-1), reduction="none")
        losses.update({"loss_iou": loss_iou.sum() / num_boxes})

        return losses


class Det3DLoss(nn.Module):
    def __init__(self, matcher, weight_dict, losses, decode_func, aux_loss=True):
        super().__init__()

        self.matcher = matcher # 匈牙利匹配损失
        self.weight_dict = weight_dict # 损失权重
        self.losses = losses # ["focal_labels", "boxes"]

        self.aux_loss = aux_loss

        self.det3d_losses = nn.ModuleDict()
        self.supervise_iou_loss = MatchabilityAwareLoss(0.25, decode_func=decode_func)
        for loss in losses:
            if loss == "boxes":
                self.det3d_losses[loss] = RegressionLoss(decode_func=decode_func)
            elif loss == "focal_labels":
                self.det3d_losses[loss] = ClassificationLoss(0.25) # alpha=0.25
            else:
                raise ValueError(f"Only boxes|focal_labels are supported for det3d losses. Found {loss}")

    @staticmethod
    def get_world_size() -> int: # 获取分布式训练时的进程总数
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    def get_target_classes(self):
        for k in self.det3d_losses.keys():
            if "labels" in k:
                return self.det3d_losses[k].src_logits, self.det3d_losses[k].target_classes

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"] # Dict 存的GT噪声组在最后一层的预测结果，同时还有前五层的结果
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"] # 3， 2 * max_gt_num * 3
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups # 2 * max_gt_num

        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def forward(self, outputs, targets, dn_meta=None, supervise_iou=False):
        '''
        outputs: Dict, DQS结果或者是模型输出的结果
        targets: List [Dict{}, Dict{}...] batch中按样本分结果
        '''
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum([len(t["labels"]) for t in targets]) # 所有样本中的GT object个数
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device) # 转张量 切换设备
        if self.get_world_size() > 1:
            torch.distributed.all_reduce(num_boxes) # 会将所有分布结果聚合，然后同步
        num_boxes = torch.clamp(num_boxes / self.get_world_size(), min=1).item()

        losses = {}

        if dn_meta is not None: # 去噪损失
            # prepare for computing denosing loss
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta) # Dict存放每一层输出结果，2 * max_gt_num 每组的gt数， 3 也即组数
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)): # batch遍历样本
                if len(targets[i]["labels"]) > 0: # 这个样本有标签
                    t = torch.arange(0, len(targets[i]["labels"]) - 1).long().cuda() # （n-1， ） BUG 0到n-2 ？ 为什么不是0 到 n-1??? 那不是最后有一个去噪丢了？不解🤔
                    t = t.unsqueeze(0).repeat(scalar, 1) # (3, n-1) 这个其实就是每一组内部的索引
                    tgt_idx = t.flatten() # （3*(n-1), ）
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t # 每一组的起始索引，加上内部索引，得到了三组中有效gt的非pad的部分的索引 （3，n-1）
                    output_idx = output_idx.flatten() #（3*(n-1), ）
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx)) # 每一组分正负样本，正样本gt的索引 分为两个轴，一个表征预测中的对应的索引，另一个则是真值gt中的索引
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx)) # 向后偏移

            l_dict = {}
            for loss in self.losses:
                l_dict.update(
                    self.det3d_losses[loss](
                        output_known_lbs_bboxes, # Dict存放每一层输出结果，但这里应该只能用到最后一层，辅助输出会在后面另行处理
                        targets, # Dict 存储gt
                        dn_pos_idx,
                        num_boxes * scalar,
                    )
                )
            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if self.aux_loss and "aux_outputs" in outputs:
            # print('outputs["aux_outputs"]', outputs["aux_outputs"])
            for i, aux_outputs in enumerate(outputs["aux_outputs"]): # 循环5次，表示前五层的输出
                indices = self.matcher(aux_outputs, targets) # 做匈牙利匹配，得到匹配结果
                for loss in self.losses:
                    l_dict = self.det3d_losses[loss](aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if dn_meta is not None:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][i]
                    l_dict = {}
                    for loss in self.losses:
                        l_dict.update(
                            self.det3d_losses[loss](
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                            )
                        )

                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets) # List[((n1),(n1)), ...] 匹配结果索引，注意，这是1000个query与gt的匹配
        if supervise_iou:
            losses.update(self.supervise_iou_loss(outputs, targets, indices, num_boxes))
            losses.update(self.det3d_losses["boxes"](outputs, targets, indices, num_boxes))
            return losses
        for loss in self.losses: # ["focal_labels", "boxes"]
            losses.update(self.det3d_losses[loss](outputs, targets, indices, num_boxes))

        return losses


def _get_src_permutation_idx(indices):
    ''''
    indices: List [((n,), (n,)), ...] 最小成本匹配索引 
    '''
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) # 标记好batch id (n1+n2+...) 形状为batch中所有的gt个数
    src_idx = torch.cat([src for (src, _) in indices]) # (n1+n2+...)  所有的gt的选择的对应的最佳匹配query的id 或者 如果是dn部分索引，则这里为（3*(n1-1) +  3*(n2-1), ....), ）也就是三组gt噪声中的正样本索引
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
