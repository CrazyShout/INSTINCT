import collections
import copy

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

from opencood.models.sub_modules.box_attention import Box3dAttention


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers # 3
        h = [hidden_dim] * (num_layers - 1) # [256, 256]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableBox3dAttention(Box3dAttention):
    def __init__(self, d_model, num_level, num_head, with_rotation=True, kernel_size=5):
        super(DeformableBox3dAttention, self).__init__(d_model=d_model, num_level=num_level, num_head=num_head,
                                                       with_rotation=with_rotation, kernel_size=kernel_size)

        self.sampling_offsets = nn.Linear(d_model, num_head * num_level * self.num_point * 2) # 预测的偏移，从 256 到 8*1*25*2

        self._create_kernel_indices(kernel_size, "kernel_indices") # 生成用于偏移的卷积核索引
        self._reset_deformable_parameters() # 初始化偏移参数

    def _reset_deformable_parameters(self): # 初始化每个头和层级的偏移参数，使不同的头能捕获不同方向和不同距离的特征
        thetas = torch.arange(self.num_head, dtype=torch.float32) * (
                2.0 * math.pi / self.num_head
        ) # thetas = [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1) # （8， 2） 每个头的角度的cos 和 sin
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0]) # 归一化最大模长为1
            .view(self.num_head, 1, 1, 2) # （8， 1， 1， 2）
            .repeat(1, self.num_level, self.num_point, 1) # （8， 1， 25， 2）
        )
        for i in range(self.num_point):
            grid_init[:, :, i, :] *= i + 1 # 这一步使不同的采样点在距离上有不同的偏移范围
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1)) # 初始化采样偏置

    def _where_to_attend(self, query, v_valid_ratios, ref_windows, h_size=188.0):
        '''
        query: self.with_pos_embed(select_src, select_pos), # (B, foreground_num, 256)  加上位置编码的前景特征
        v_valid_ratios: None
        ref_windows: (B, foreground_num, 7) 参考窗口
        h_size: H*W 开根号
        '''
        B, L = ref_windows.shape[:2]

        offset_boxes = self.linear_box(query) # (B, foreground_num, 256) --> (B, foreground_num, 1*8*4) 如果带旋转的话是1*8*5
        offset_boxes = offset_boxes.view(B, L, self.num_head, self.num_level, self.num_variable) # (B, foreground_num, 8， 1，4)  这个是在算 boxAttention中的box大小

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3) # (B, foreground_num, 1, 1, 7)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        ref_boxes = ref_windows[..., [0, 1, 3, 4]] # 中心点坐标xy以及长宽hw (B, foreground_num, 1, 1, 4)
        ref_angles = ref_windows[..., [6]] # 角度 

        if self.with_rotation:
            offset_boxes, offset_angles = offset_boxes.split(4, dim=-1)
            angles = (ref_angles + offset_angles / 16) * 2 * math.pi
        else:
            angles = ref_angles.expand(B, L, self.num_head, self.num_level, 1) # 角度不变  (B, foreground_num, 8, 1, 1)

        boxes = ref_boxes + offset_boxes / 8 * ref_boxes[..., [2, 3, 2, 3]] # 参考窗口加上偏移 offset_boxes预测了四维度偏移，包括中心点与长宽，除以注意力头数均衡每个注意头预测的结果  (B, foreground_num, 8， 1，4)
        center, size = boxes.unsqueeze(-2).split(2, dim=-1) # 分成中心和尺度  都是(B, foreground_num, 8， 1，1，2)

        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        rot_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1) # 旋转矩阵 (B, foreground_num, 8, 1, 1， 4)
        rot_matrix = rot_matrix.view(B, L, self.num_head, self.num_level, 1, 2, 2) # (B, foreground_num, 8, 1, 1, 2, 2)

        sampling_offsets = self.sampling_offsets(query).view( # (B, foreground_num, 256)  加上位置编码的前景特征 线性变化为 (B, foreground_num, 8*1*25*2)，最后变形为 (B, foreground_num, 8, 1, 25, 2)
            B, L, self.num_head, self.num_level, self.num_point, 2
        )
        deformable_grid = sampling_offsets / h_size # 归一化 最终这个可变形的grid其实就是seed DGA中相比BoxAttention 唯一的创新 这是通过让采样点也动态活动起来实现的
        # kernel_indices其实记录着25个采样点的标准化采样位置，接下来映射到参考框对应的位置
        fixed_grid = self.kernel_indices * torch.relu(size) # （5x5， 2） * (B, foreground_num, 8， 1，1，2) = (B, foreground_num, 8， 1，5x5，2)
        fixed_grid = center + (fixed_grid.unsqueeze(-2) * rot_matrix).sum(-1) # 加式右边是 (B, foreground_num, 8， 1，5x5，1，2) * (B, foreground_num, 8, 1, 1, 2, 2) . sum(-1)= (B, foreground_num, 8， 1，25，2)

        grid = fixed_grid + deformable_grid

        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()


class SEEDTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            nlevel=4,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            code_size=7,
            num_queries=300,
            keep_ratio=0.5,
            iou_rectifier=None,
            iou_cls=None,
            cp_flag=False,
            num_classes=3,
            mom=0.999,
    ):
        super().__init__()
        self.num_queries = num_queries # 1000
        self.num_classes = num_classes # 1
        self.code_size = code_size # 7
        self.iou_rectifier = iou_rectifier # [ 0.68, 0.71, 0.65 ]
        self.iou_cls = iou_cls # [0, 1]
        self.m = mom # 0.999

        self.dga_layer = DGALayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation, keep_ratio)

        decoder_layer = SEEDDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation,
                                         code_size=code_size, num_classes=num_classes)
        self.decoder = SEEDDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)

    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), # 形成坐标网格 两个的形状都是 （H， W）
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H # 归一化坐标 且在最前方增加一维，(1, H*W)
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1) # （1， H*W， 2）
            ref_wh = torch.ones_like(ref_xy) * 0.025 # (1, H * W, 2) 全0.025
            placeholder = torch.zeros_like(ref_xy)[..., :1] # (1, H * W, 1) 初始化占位符张量为全0
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1) #  (1, H * W, 7) 最后一维度分别为，归一化的x，y坐标，中心点位置初始化为0.5， 参考窗口的宽高，另一个占位符，占位符
            extra = self.code_size - ref_box.shape[-1] # 7-7=0
            ref_box = torch.cat([ref_box, torch.zeros_like(ref_box[..., 0:extra])], dim=-1) # 补充至 (1, H * W, code_size) 因为code size就是7 所以无需补充
            ref_box = ref_box.expand(B, -1, -1) # (B, H * W, code_size)
            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1) # (B, total_windows, code_size) == (B, H * W, code_size)
        return ref_windows

    def _quality_query_selection(self, enc_embed, ref_windows, indexes=None):
        '''
        enc_embed: (B, H*W, 256) 包含粗查询的 query 特征
        ref_windows: (B, H * W, 7)     
        '''
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows, out_ious = self.proposal_head(enc_embed, ref_windows) # 生成proposal 分为 分类logits(B, H * W, 1)  、boxes 定位 (B, H * W, 7) 、 IoU预测 (B, H * W, 1) 

        out_logits_max, out_labels = out_logits.max(dim=-1, keepdim=True) # 选出最大logit (B, H * W, 1) , 第二项是最大项索引，0 1 2三种，刚好作为标签
        out_probs = out_logits_max[..., 0].sigmoid() # 归一化成为概率 (B, H * W, ) 
        out_labels = out_labels[..., 0] # (B, H * W, ) 

        mask = torch.ones_like(out_probs) # (B, H * W, ) 全1
        for i in range(out_logits.shape[-1]): # 最后一维度为1，表示类别无关
            if i not in self.iou_cls: # [0, 1] 这里遍历到i=2的时候会成功进入， 如果是类别无关，这里其实没用，这本来是遮蔽掉不想关注的类别
                mask[out_labels == i] = 0.0 # 所有最大值在索引2处取的都置为0

        score_mask = out_probs > 0.3 # (B, H * W, )  大于阈值的全部置为True
        mask = mask * score_mask.type_as(mask) # 再将Cyclist对应的部分mask掉，这是因为这种类别不均衡切置信度一般不高，可能引入噪声会影响IoU校准

        if isinstance(self.iou_rectifier, list): # [ 0.68, 0.71, 0.65 ]
            out_ious = (out_ious + 1) / 2 # iou得分缩放到0-1
            iou_rectifier = torch.tensor(self.iou_rectifier).to(out_probs)
            temp_probs = torch.pow(out_probs, 1 - iou_rectifier[out_labels]) * torch.pow(
                out_ious[..., 0], iou_rectifier[out_labels])
            out_probs = out_probs * (1 - mask) + mask * temp_probs # 这里对应的论文中的公式3

        elif isinstance(self.iou_rectifier, float): # 类别无关会执行以下分支
            out_ious = (out_ious + 1) / 2 # iou得分缩放到0-1
            temp_probs = torch.pow(out_probs, 1 - self.iou_rectifier) * torch.pow(out_ious[..., 0], self.iou_rectifier)
            out_probs = out_probs * (1 - mask) + mask * temp_probs # (B, H * W, ) 
        else:
            raise TypeError('only list or float')
        #

        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False)# 选1000个最高的  (B, 1000, ) 
        indexes = indexes.unsqueeze(-1) # (B, 1000, 1) 

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1])) # 抽取出对应的box (B, 1000, 7) 
        topk_probs_class = torch.gather(out_logits, 1, indexes.expand(-1, -1, out_logits.shape[-1])) # 抽取出 conf (B, 1000, 1) 
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs_class.sigmoid().detach(),
            ),
            dim=-1,
        ) # (B, 1000, 8) 

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()): # 后者是前者的深拷贝，不进行梯度计算
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m) # 可能用于对比学习，辅助模型参数=辅助模型参数x动量系数 + 主模型参数x(1-动量系数)

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None,
                score_mask=None):
        '''
        src: [(B, 256, H, W)]
        pos: [(B, 256, H, W)]
        noised_gt_box: (B, pad_size, 7)
        noised_gt_onehot: (B, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # (B, H * W, 7）  
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            b, c = pe.shape[:2]
            pe = pe.view(b, c, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)
        score_mask = score_mask.flatten(-2) # (B, H*W)

        coarse_memory = self.dga_layer(src, src_pos, src_shape, src_start_index, src_anchors, score_mask) # (B, H*W, 256) 这里只计算了选择好的query进行DGA，然后放回去，这里其实就是前景选择出粗query，然后送入了dga
        query_embed, query_pos, topk_proposals, topk_indexes = self._quality_query_selection(coarse_memory, src_anchors) # 返回None，None，(B, 1000, 8)，(B, 1000, 1)  最后两个是通过质量评分来选择的query 注意，8维度中最后1维存的是分类得分

        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ), # type: ignore
                dim=-1,
            ) # 合并在一起 (B, pad_size, 8) 前面7个是box，后面1个是class label
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            ) # (B, pad_size + 1000, 8)

        init_reference_out = topk_proposals[..., :self.code_size] # (B, pad_size + 1000, 7) dqs得到的初始 query 1000个

        hs, inter_references = self.decoder(query_embed, query_pos, coarse_memory, src_shape, src_start_index,
                                            topk_proposals, attn_mask) # (6, B, pad_size + 1000, 256) 每一层的输出的query特征， (6， B, pad_size + 1000, 7) 每一层的检测结果

        # optional gt forward
        if targets is not None:
            batch_size = len(targets)
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # 每个样本中对应的gt的个数 [n1, n2]
            max_gt_num = max(per_gt_num) # max值
            batched_gt_boxes_with_score = coarse_memory.new_zeros(batch_size, max_gt_num, self.code_size + self.num_classes) # （B，max_gt_num，8）
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :self.code_size] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], self.code_size:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // max((max_gt_num * 2), 1) # 得到去噪gt组数 == 3  每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2)) # [0, 2, 4] 组索引 一共 6*max_gt_num 个gt
                    pos_noised_gt_proposals = torch.cat( 
                        [noised_gt_proposals[:, pi * max_gt_num: (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    )# 每一组抽取max_gt_num个 (B, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1) # (B, 4*max_gt_num, 8) 三批gt噪声正样本前放一批gt
                    # create attn_mask for gt groups
                    gt_attn_mask = coarse_memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool() # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                        di * max_gt_num: (di + 1) * max_gt_num,
                        di * max_gt_num: (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (6，B, 4*max_gt_num, 256) 与 (6，B, 4*max_gt_num, 8)
                    None,
                    None,
                    coarse_memory,
                    src_shape,
                    src_start_index,
                    gt_proposals, # (B, 4*max_gt_num, 8) 这个会作为query使用
                    gt_attn_mask,
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :self.code_size],
                ),
                dim=1,
            ) # (B, pad_size + 1000 + 4*max_gt_num, 7) 6批噪声gt+初始的dqs得到的1000个box 加上 4批 gt(第1批是gt，后面3批示噪声gt正样本)

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (6， B, pad_size + 1000 + 4*max_gt_num, 256) pad_size其实等于 6*max_gt_num
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (6， B, pad_size + 1000 + 4*max_gt_num, 8) pad_size其实等于 6*max_gt_num 每一层的预测结果

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out, coarse_memory, src_anchors, topk_indexes


class DGALayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation, keep_ratio):
        super().__init__()
        self.keep_ratio = keep_ratio #  # 0.3
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True) # 多头自注意力 特征维度是256， 注意力头数是8, 这里缺少batch_first设置，已经提Issue，作者已采纳 --xyj
        self.cross_attn = DeformableBox3dAttention(d_model, nlevel, nhead, with_rotation=False) # 可变形 box 注意力 nlevel = 1
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, src_shape, src_start_idx, ref_windows, score_mask):
        '''
        预测前景query, 前景query生成采样grid, 随后
        src: (B, H*W, 256)
        pos: (B, H*W, 256) 位置编码
        src_shape: (1, 2) 记录着特征图形状
        src_start_idx: 划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)
        ref_windows: (B, H * W, 7)
        score_mask:  (B, H*W) 预测出来的mask
        '''
        foreground_num = math.ceil(score_mask.shape[1] * self.keep_ratio) # 保留30%的数量
        select_score_mask, indices = torch.topk(score_mask, k=foreground_num, dim=-1) #  两者形状都是(B, foreground_num)
        select_src = torch.gather(src, 1, indices.unsqueeze(-1).repeat(1, 1, src.size(-1))) # (B, foreground_num, 256)  前景特征
        select_pos = torch.gather(pos, 1, indices.unsqueeze(-1).repeat(1, 1, pos.size(-1)))
        select_ref_windows = torch.gather(ref_windows, 1, indices.unsqueeze(-1).repeat(1, 1, ref_windows.size(-1))) # (B, foreground_num, 7)

        query_num = math.ceil(foreground_num * self.keep_ratio) # 进一步缩减一下
        query_indices = torch.topk(select_score_mask, k=query_num, dim=-1).indices
        query_src = torch.gather(select_src, 1, query_indices.unsqueeze(-1).repeat(1, 1, select_src.size(-1))) # (B, query_num, 256)
        query_pos = torch.gather(select_pos, 1, query_indices.unsqueeze(-1).repeat(1, 1, select_pos.size(-1))) # (B, query_num, 256)

        q = k = self.with_pos_embed(query_src, query_pos) # 两者相加 (B, query_num, 256)
        query_src2 = self.self_attn(q, k, query_src)[0] # 输出结果为(B, query_num, 256)，和 (B, query_num, query_num) 后者是注意力权重，所以暂时不需要
        query_src = query_src + query_src2 # 残差链接
        query_src = self.query_norm(query_src)
        select_src = select_src.scatter(1, query_indices.unsqueeze(-1).repeat(1, 1, select_src.size(-1)), query_src) # 做完自注意力的query 特征写回前景特征

        output = src # 这是一开始的展平特征  (B, H*W, 256)

        src2 = self.cross_attn(
            self.with_pos_embed(select_src, select_pos), # (B, foreground_num, 256)  加上位置编码的前景特征
            src, #  (B, H*W, 256)
            src_shape, # (1, 2) 记录着特征图形状
            None,
            src_start_idx, # 划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)
            None,
            select_ref_windows, # (B, foreground_num, 7) 参考窗口 BoxAttention需要的
        ) # 返回列表，两个元素分别为 DGA结果：(B, foreground_num, 256) 和 DGA采样权重: (B, foreground_num, 8, 1, 5, 5) 分为8个头，每个头负责32dimension

        select_src = select_src + self.dropout1(src2[0]) # (B, foreground_num, 256) 残差连接
        select_src = self.norm1(select_src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(select_src)))) # 256-->1024-->256
        select_src = select_src + self.dropout2(src2)
        select_src = self.norm2(select_src)

        output = output.scatter(1, indices.unsqueeze(-1).repeat(1, 1, output.size(-1)), select_src) # 放回去 (B, H*W, 256)

        return output


class SEEDDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation, code_size=7, num_classes=3):
        super().__init__()
        self.code_size = code_size
        self.num_classes = num_classes

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = DeformableBox3dAttention(d_model, nlevel, nhead, with_rotation=True)

        self.pos_embed_layer = MLP(code_size + num_classes, d_model, d_model, 3)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, idx, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None):
        '''
        idx: 当前层编号, 一共6层 0-5
        query: None 一开始是None, 在第一层后每层都会更新为上一层输出的query
        query_pos: None
        memory: (B, H*W, 256) 包含了粗query 
        memory_shape: (1, 2) 记录着特征图形状[[H, W]]
        memory_start_idx: (0, ) 生成的split索引, 由于只有一个尺度， 所以这里就是0
        ref_windows: (B, pad_size + 1000, 10) 包含了噪声GT以及 dqs输出的1000个query  后续每一层会逐渐输出检测结果, 从而更新
        attn_mask: (1000+pad_size, 1000+pad_size)
        
        '''
        if idx == 0: # 第一层
            query = self.pos_embed_layer(ref_windows) # (B, pad_size + 1000, 8) --> (B, pad_size + 1000, 256)
            q = k = query
        elif query_pos is None:
            query_pos = self.pos_embed_layer(ref_windows) # 用前面一层生成的预测作为输入，生成位置编码，(B, pad_size + 1000, 8) --> (B, pad_size + 1000, 256)
            q = k = self.with_pos_embed(query, query_pos)

        q = q.transpose(0, 1) # (pad_size + 1000, B, 256)
        k = k.transpose(0, 1) # (pad_size + 1000, B, 256)
        v = query.transpose(0, 1) # (pad_size + 1000, B, 256)

        query2 = self.self_attn(q, k, v, attn_mask=attn_mask)[0] # 做自注意力
        query2 = query2.transpose(0, 1) # (B, pad_size + 1000, 256) 
        query = query + self.dropout1(query2) # 残差
        query = self.norm1(query)

        query2 = self.cross_attn( # 所有的query去memory中采样
            self.with_pos_embed(query, query_pos),
            memory,
            memory_shape,
            None,
            memory_start_idx,
            None,
            ref_windows[..., :self.code_size],
        )[0] # (B, pad_size + 1000, 256) 和 (B, pad_size + 1000, 8, 1, 5, 5) 第二个返回就是25个点的采样权重，8个头每个负责32维度

        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query


class SEEDDecoder(nn.Module):
    def __init__(self, d_model, decoder_layer, num_layers, cp_flag):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers) # 克隆6次
        self.cp_flag = cp_flag # True
        self.code_size = decoder_layer.code_size # 7

    def forward(self, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None):
        '''
        query: None
        query_pos: None
        memory: (B, H*W, 256) 包含了粗query
        memory_shape: (1, 2) 记录着特征图形状[[H, W]]
        memory_start_idx: (0, ) 生成的split索引, 由于只有一个尺度， 所以这里就是0
        ref_windows: (B, pad_size + 1000, 10) 包含了噪声GT以及 dqs输出的1000个query
        attn_mask: (1000+pad_size, 1000+pad_size)
        '''
        output = query
        intermediate = []
        intermediate_ref_windows = []

        for idx, layer in enumerate(self.layers):
            if self.cp_flag: # 优化内存和加速训练，一旦启用就会逐层重计算，会选择性地保存，然后后续重新计算，所以可以节省内存消耗
                output = cp.checkpoint(layer, idx, output, query_pos, memory, memory_shape, memory_start_idx,
                                       ref_windows, attn_mask) # (B, pad_size + 1000, 256)
            else:
                output = layer(idx, output, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask) # (B, pad_size + 1000, 256)
            new_ref_logits, new_ref_windows, new_ref_ious = self.detection_head(output, ref_windows[..., :self.code_size], idx) # 检测一遍结果
            new_ref_probs = new_ref_logits.sigmoid()  # .max(dim=-1, keepdim=True).values
            ref_windows = torch.cat(
                (
                    new_ref_windows.detach(),
                    new_ref_probs.detach(),
                ),
                dim=-1,
            ) # (B, pad_size + 1000, 8) 这一层的预测结果会作为下一层的参考框，这样层层递进给模型逐渐逼近结果的机会
            intermediate.append(output) # 存放每一层输出的query (B, pad_size + 1000, 256)
            intermediate_ref_windows.append(new_ref_windows) # 存放每一层输出的检测结果 (B, pad_size + 1000, 8) 
        return torch.stack(intermediate), torch.stack(intermediate_ref_windows) # （6，B, pad_size + 1000, 256）  （6，B, pad_size + 1000, 8）


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def flatten_with_shape(tensor_list, mask_list):
    """
    Params:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]

    Return:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)
    """
    assert isinstance(tensor_list, collections.abc.Sequence)
    assert len(tensor_list) > 0

    N = len(tensor_list) # 1
    tensor_shape = torch.zeros(N, 2, dtype=torch.int64, device=tensor_list[0].device) # (1, 2)
    tensor_flatten = []

    if mask_list is not None:
        mask_flatten = []

    for i, tensor in enumerate(tensor_list):
        new_tensor = tensor.flatten(2).permute(0, 2, 1) # 展平成（B，H*W，C）
        tensor_flatten.append(new_tensor)

        if mask_list is not None:
            mask = mask_list[i]
            new_mask = mask.flatten(1)
            mask_flatten.append(new_mask)
            assert tensor.shape[2] == mask.shape[1]
            assert tensor.shape[3] == mask.shape[2]
        tensor_shape[i, 0] = tensor.shape[2]
        tensor_shape[i, 1] = tensor.shape[3]

    mask_flatten = torch.cat(mask_flatten, dim=1) if mask_list is not None else None
    tensor_flatten = torch.cat(tensor_flatten, dim=1)

    return tensor_flatten, mask_flatten, tensor_shape
