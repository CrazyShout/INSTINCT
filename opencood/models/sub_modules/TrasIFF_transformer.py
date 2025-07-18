import collections
import copy

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import numpy as np
import math
from collections import defaultdict

from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils
from opencood.models.sub_modules.box_attention import Box3dAttention
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils import box_utils
from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_utils
from scipy.optimize import linear_sum_assignment
from opencood.models.comm_modules.target_assigner.hungarian_assigner import HungarianMatcher3d, generalized_box3d_iou, \
    box_cxcyczlwh_to_xyxyxy
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
    
class FeatureMagnet(nn.Module):
    def __init__(self, d_model=256, nhead=8,  dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()

        self.query_fusion = MLP(d_model*2, d_model, d_model, 3)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
    
    def forward(self, x, pos_1d, ego_idx=0, ref_windows=None):
        '''
        x: [(1, L1, C), (1, L2, C)...] ego和其他agent的query, 这些query已经被CDA模块处理过了
        pos_1d: [(1, L1, 1), (1, L2, 1)...] 标记了在统一坐标系下每个对应query的位置, 这个位置具有唯一性 但是由于其他agent旋转后取整, 可能造成1d编码相同
        ref_windows: [(1, L1, 7), (1, L2, 7)...]  # 参考框, 用来BoxAttention
        '''
        B_N = x[0].shape[0]  # Batch size
        final_queries = []  # To hold final queries
        
        # Process each agent and the ego agent separately
        ego_queries = x[ego_idx] # (1, L1, C)
        ego_positions = pos_1d[ego_idx] # (1, L1, 1)
        
        # Collect all other agents' queries and positions
        other_agents_queries = x[:ego_idx] + x[ego_idx+1:]
        other_agents_positions = pos_1d[:ego_idx] + pos_1d[ego_idx+1:]
        
        # 1. For the ego agent, we want to collect the queries based on position matching
        ego_position_queries = {}  # Dictionary to hold queries by position
        for i in range(ego_queries.shape[1]): # 遍历每个向量
            pos = ego_positions[0, i].item()  # Get the position (assuming batch_size=1 for simplicity)
            if pos not in ego_position_queries:
                ego_position_queries[pos] = []
            ego_position_queries[pos].append(ego_queries[:, i, :]) # pos和feature 是对应的，这里直接将(1,C)放入
        
        # 2. For each other agent, we process its queries based on the positions
        for agent_queries, agent_positions in zip(other_agents_queries, other_agents_positions):
            agent_position_queries = {}  # Dictionary to hold queries by position
            for i in range(agent_queries.shape[1]):
                pos = agent_positions[0, i].item()  # Get the position
                if pos not in agent_position_queries:
                    agent_position_queries[pos] = []
                agent_position_queries[pos].append(agent_queries[:, i, :]) # 同一个位置的放入一个列表里，注意这里可能由于旋转造成位置重叠
            
            # 3. Fuse the agent's queries with the ego's queries at the same position
            for pos, agent_query_list in list(agent_position_queries.items()): # list创建副本 
                if pos in ego_position_queries: # 如果在ego中也有重复的位置 
                    # Fuse the queries using MLP (aggregation of queries at the same position)
                    sum_tensor = sum(agent_query_list) # 以防扭曲后的重叠投影，位置相同的就直接相加在一起 TODO 直接相加是否合理？如果用坐标网格来实现TransIFF？
                    ego_position_queries[pos] += [sum_tensor] # [(1,C), (1,C)]

                    agent_position_queries.pop(pos)

                    # agent_queries_fused = torch.cat(agent_query_list, dim=1)  # Concatenate queries for the same position
                    ego_queries_fused = torch.cat(ego_position_queries[pos], dim=1)  # Concatenate ego queries for the same position (1, C*2)
                    # Perform fusion
                    fused_queries = self.query_fusion(ego_queries_fused)  # MLP fusion (1, C)
                    # Update the ego's query with the fused result
                    ego_position_queries[pos] = [fused_queries]  # Update with the fused queries
                else: # 如果没有重叠也要保证所有投影后的位置唯一，重叠的query就相加在一起
                    sum_tensor = sum(agent_query_list)
                    agent_position_queries[pos] = [sum_tensor]
        
        # Now combine the final queries for ego and other agents 重新恢复成 (1, l1, C)
        final_ego_queries = torch.cat([val[0] for val in ego_position_queries.values()], dim=0).unsqueeze(0)  # Concatenate all position-based fused queries
        final_queries.append(final_ego_queries)  # Add Ego's final queries

        agent_remain_queries = torch.cat([val[0] for val in agent_position_queries.values()], dim=0).unsqueeze(0) # (1, l2', C)
        final_queries.append(agent_remain_queries)  # Add Ego's final queries

        # Stack all the final queries together
        final_queries = torch.cat(final_queries, dim=1)  # Concatenate queries from ego and agents (1, l1+l2', C) 这个也就是paper中提到的 optimal Q
        # print("final_queries shape is ", final_queries.shape)

        k = v =  torch.cat(x, dim=1) # 原始的 所有agent 筛选的query feature 全部concat 在一起，形成(1, l1+l2, C)
        # print("k shape is ", k.shape)

        outputs = self.self_attn(final_queries, k, v)[0]
        final_queries = final_queries + self.dropout(outputs)
        final_queries = self.norm1(final_queries)
        outputs = self.linear2(self.dropout(self.activation(self.linear1(final_queries))))
        final_queries = final_queries + self.dropout(outputs)
        final_queries = self.norm2(final_queries)

        return final_queries # (1, L, C) 最终就是输出这个L个query


class CrossDomainAdaption(nn.Module):
    def __init__(self, d_model=256, nhead=8,  dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn_ego = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_inf = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ego_linear1 = nn.Linear(d_model, dim_feedforward)
        self.ego_linear2 = nn.Linear(dim_feedforward, d_model)
        self.ego_norm1 = nn.LayerNorm(d_model)
        self.ego_norm2 = nn.LayerNorm(d_model)

        self.inf_linear1 = nn.Linear(d_model, dim_feedforward)
        self.inf_linear2 = nn.Linear(dim_feedforward, d_model)
        self.inf_norm1 = nn.LayerNorm(d_model)
        self.inf_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, pos):
        '''
        x: [(1, L1, C), (1, L2, C)...] ego和其他agent的筛选过的query ego有L1个 agent1有L2个 以此类推
        pos: [(1, L1, C), (1, L2, C)...] 位置编码, 用的是空间变换过的位置编码
        首先是加位置编码, 加了位置编码后concat在一起形成
        '''
        features = []
        for i, feat in enumerate(x):
            feat = self.with_pos_embed(feat, pos[i])
            features.append(feat)
        k =v = torch.cat(features, dim=1) # (1, L1+L2+..., C)
        assert len(x) == 2 # TransIFF 是专门面向V2I设计
        q_ego, q_inf = features[0], features[1]

        # two-stream
        q_ego_2 = self.self_attn_ego(q_ego, k, v)[0]
        q_ego = q_ego + self.dropout(q_ego_2)
        q_ego = self.ego_norm1(q_ego)
        q_ego_2 = self.ego_linear2(self.dropout(self.activation(self.ego_linear1(q_ego))))
        q_ego = q_ego + self.dropout(q_ego_2)
        q_ego = self.ego_norm2(q_ego)

        q_inf_2 = self.self_attn_inf(q_inf, k, v)[0]
        q_inf = q_inf + self.dropout(q_inf_2)
        q_inf = self.inf_norm1(q_inf)
        q_inf_2 = self.inf_linear2(self.dropout(self.activation(self.inf_linear1(q_inf))))
        q_inf = q_inf + self.dropout(q_inf_2)
        q_inf = self.ego_norm2(q_inf)

        return [q_ego, q_inf]


class TransIFFEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        # self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers)

    def _generate_relative_position_encoding(self, H, W):
        # 创建一个网格，其中每个位置的 (i, j) 坐标
        grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
        
        # 将位置编码堆叠成 (H, W, 2) 的形状
        position_encoding = torch.stack((grid_x, grid_y), dim=-1)
        
        return position_encoding

    def _transform_position_matrix(self, position_matrix, transform_matrix):
        # position_matrix 是 H x W x 2 的相对位置矩阵
        # transform_matrix 是 4 x 4 的变换矩阵
        
        H, W, _ = position_matrix.shape
        transformed_positions = torch.zeros_like(position_matrix)
        
        # 对每个位置进行变换
        for i in range(H):
            for j in range(W):
                # 原始相对位置坐标 (x, y)
                x, y = position_matrix[i, j]
                
                # 将 (x, y) 转换为齐次坐标 (x, y, 0, 1)
                original_coords = torch.tensor([x, y, 0, 1.0])  # 齐次坐标
                
                # 应用变换矩阵
                transformed_coords = torch.matmul(transform_matrix, original_coords.float())
                
                # 获取变换后的坐标（不再需要齐次坐标）
                transformed_positions[i, j] = transformed_coords[:2]  # 只取 x, y
        
        return transformed_positions


    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            ) # 两个shape 都是(H,W)

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1) # (HW,2)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None):
        '''
        这个函数主要是负责筛选出合适的query, 也就是query选择. 同时要将位置编码进行旋转， 统一到ego的位置编码
        '''
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows) # 生成proposal 分为 分类logits(B, H * W, 1)  、boxes 定位 (B, H * W, 7)

        out_probs = out_logits[..., 0].sigmoid()
        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False) # 选出置信度较高的
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None):
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src)
        src, _, src_shape = flatten_with_shape(src, None)
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # (B, HW, C)
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这个是用于一次处理多个尺度的feature的，在我们这里就是(0,)

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors) # (B, H*W, 256) 通过BoxAttention进行了交互
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors)# 返回None，None，(B, query_num, 8)，(B, query_num, 1)

        select_memory = torch.gather(memory, 1, topk_indexes.expand(-1, -1, memory.shape[-1])) # (B, query_num, 256)

        return select_memory, topk_proposals, None, memory, src_anchors, topk_indexes
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            )
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            )
        init_reference_out = topk_proposals[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references = self.decoder(
            query_embed,
            query_pos,
            memory,
            src_shape,
            src_start_index,
            topk_proposals,
            attn_mask,
        )

        # optional gt forward
        if targets is not None:
            batch_size = len(targets)
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets]
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"]
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder()
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2)
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    )
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()
                    for di in range(dn_group_num + 1):
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt(
                    None,
                    None,
                    memory,
                    src_shape,
                    src_start_index,
                    gt_proposals,
                    gt_attn_mask,
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            )

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            )
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            )

        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out, memory, src_anchors, topk_indexes

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0, keepdim=True)[0]

# simple fusion use Scaled Dot Product Attention
class AttenQueryFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenQueryFusion, self).__init__()
        self.sqrt_dim = np.sqrt(feature_dim)

    def forward(self, x):
        # x : 1, k, C
        if x.size(1) == 1:
            return x
        query = key = value = x
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value) # (1,k,C)
        x = context[:,0:1,:]
        return x

class SimpleGatingFusion(nn.Module):
    """
    对特征进行融合的模块:
    输入:
        x: (1, k, C) 同一位置k个Agent特征, k<=4
    实现:
        1. Flatten成为 (1, k*C)
        2. 通过MLP输出(k,)维权重向量
        3. 使用softmax归一化后加权求和
    """
    def __init__(self, d_model=256, max_agents=2, hidden_dim=128):
        super(SimpleGatingFusion, self).__init__()
        self.max_agents = max_agents
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model * max_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_agents)
        )

    def forward(self, x):
        # x: (1, k, C), k <= self.max_agents
        k = x.size(1)
        if k == 0:
            # 没有特征可融合，直接返回空
            return x
        if k == 1:
            # 只有一个特征，直接返回
            return x

        # 如果实际k < max_agents，用0填充
        pad_num = self.max_agents - k
        if pad_num > 0:
            pad = x.new_zeros((1, pad_num, self.d_model))
            x_padded = torch.cat([x, pad], dim=1) # (1, max_agents, C)
        else:
            x_padded = x

        flattened = x_padded.view(1, -1)  # (1, max_agents*C)
        weights = self.mlp(flattened)     # (1, max_agents)
        weights = weights[:, :k]          # 只取前k个权重
        weights = F.softmax(weights, dim=-1)  # (1, k)
        
        # 加权求和
        fused = torch.sum(x * weights.unsqueeze(-1), dim=1, keepdim=True) # (1,1,C)
        return fused

class BoxGatingFusion(nn.Module):
    """
    对参考框进行融合的模块:
    输入:
        boxes: (1, k, 8) 同一位置k个Agent提供的box参数, k<=4
    实现:
        与特征类似，用MLP对k个box打分并加权平均。
    假设8维box为 [cx, cy, cz, w, l, h, rot, score]
    """
    def __init__(self, box_dim=8, max_agents=2, hidden_dim=64):
        super(BoxGatingFusion, self).__init__()
        self.max_agents = max_agents
        self.box_dim = box_dim
        self.mlp = nn.Sequential(
            nn.Linear(box_dim * max_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_agents)
        )

    def forward(self, boxes):
        # boxes: (1, k, 8)
        k = boxes.size(1)
        if k == 0:
            return boxes
        if k == 1:
            return boxes

        pad_num = self.max_agents - k
        if pad_num > 0:
            pad = boxes.new_zeros((1, pad_num, self.box_dim))
            boxes_padded = torch.cat([boxes, pad], dim=1) # (1, max_agents, 8)
        else:
            boxes_padded = boxes

        flattened = boxes_padded.view(1, -1) # (1, max_agents*8)
        weights = self.mlp(flattened) # (1, max_agents)
        weights = weights[:, :k] 
        weights = F.softmax(weights, dim=-1) # (1, k)

        # 取空间参数7维加权平均，再对score维度也加权平均
        space_params = boxes[..., :7] # (1,k,7)
        fused_space = torch.sum(space_params * weights.unsqueeze(-1), dim=1, keepdim=True) # (1,1,7)

        score = boxes[..., 7:8] # (1,k,1)
        fused_score = torch.sum(score * weights.unsqueeze(-1), dim=1, keepdim=True) # (1,1,1)

        fused_box = torch.cat([fused_space, fused_score], dim=-1) # (1,1,8)
        return fused_box


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
        box_encode_func=None, 
        box_decode_func=None, 
        get_sparse_features_func=None
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom
        self.extra_query_num = 200 # 额外的query数量，用于非重叠位置的补充

        self.box_encode_func=box_encode_func
        self.box_decode_func=box_decode_func
        self.get_sparse_features_func=get_sparse_features_func

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        self.trans_adapter = TransAdapt(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        # self.query_fusion = AttenQueryFusion(d_model)
        # self.ref_fusion = AttenQueryFusion(8)
        self.query_fusion = SimpleGatingFusion()
        self.ref_fusion = BoxGatingFusion()
        self.foreground_fusion = MaxFusion()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)
        self.sample_idx = 0

    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

        out_probs = out_logits[..., 0].sigmoid()
        topk_probs, indexes = torch.topk(out_probs, self.num_queries + self.extra_query_num, dim=1, sorted=True)
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)
        # print("out_probs  is ", [round(x, 3) for x in out_probs[0][:1000].tolist()])

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, valid_bboxes_single = None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None):
        '''
        src: [(B_n, 256, H, W)]
        pos: [(B_n, 256, H, W)]
        noised_gt_box: (B, pad_size, 7)  这里用的应该是协同gt
        noised_gt_onehot: (B, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # 创造参考框，这个是BoxAttention必须的 (B_n, HW, 7)
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B_n, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B_n, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors) # BoxAttention 提取特征 结果为(B_n, H*W, 256)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors) # 返回None，None，(B_n, query_num+extra_num, 8)，(B_n, query_num+extra_num, 1)
        
        ego_topk_proposals = topk_proposals[:, :self.num_queries, :] # (B_n, query_num, 8)
        ego_topk_indexes = topk_indexes[:, :self.num_queries, :] # (B_n, query_num, 1) NOTE single监督只监督前300个
        extra_topk_proposals = topk_proposals[:, self.num_queries:, :]  # (B_n, extra_num, 8)
        extra_topk_indexes = topk_indexes[:, self.num_queries:, :]  # (B_n, extra_num, 1)

        fined_query = torch.gather(memory, 1, ego_topk_indexes.expand(-1, -1, memory.shape[-1])) # (B_n, query_num, C) refine的query
        extra_query = torch.gather(memory, 1, extra_topk_indexes.expand(-1, -1, memory.shape[-1])) # (B_n, extra_num, C) refine的query

        H, W = src_shape[0,0], src_shape[0,1]
        memory_discrete = torch.zeros_like(memory) # (B_n, H*W, 256) 
        memory_discrete = memory_discrete.scatter(1, ego_topk_indexes.repeat(1, 1, memory_discrete.size(-1)), fined_query) # (B_n, H*W, 256) 将query放入到一个空的memory中
        memory_discrete = memory_discrete.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W) # (B_n, C, H, W) 形成稀疏的特征图

        # 新建一个默认参考框，然后将encoder预测的内容填充进去，这个将会在空间变换后作为
        ref_boxes_before_trans = copy.deepcopy(src_anchors)
        ref_probs_before_trans = torch.zeros(ref_boxes_before_trans.size(0), ref_boxes_before_trans.size(1), 1).to(ref_boxes_before_trans)
        ref_boxes_before_trans = torch.cat([ref_boxes_before_trans, ref_probs_before_trans], dim=-1)
        fined_ref_boxes = ego_topk_proposals # (B_n, query_num, 8) 这个是参考框 要跟着采样
        ref_boxes_before_trans = ref_boxes_before_trans.scatter(1, ego_topk_indexes.repeat(1, 1, ref_boxes_before_trans.size(-1)), fined_ref_boxes) # (B_n, H*W, 8) 将query放入到一个空的memory中

        ref_boxes_before_trans = ref_boxes_before_trans.permute(0, 2, 1).reshape(memory.shape[0], 8, H, W) # (B_n, 8, H, W) 形成稀疏的特征图

        # 创造mask标记fined query
        valid_flag = torch.ones(fined_query.shape[0], fined_query.shape[1], 1).to(fined_query) # (B_n, query_num, 1) 全1
        memory_mask = torch.zeros(memory.shape[0], memory.shape[1], 1).to(memory) # (B_n, HW, 1)
        memory_mask = memory_mask.scatter(1, ego_topk_indexes.repeat(1, 1, memory_mask.size(-1)), valid_flag) # (B_n, HW, 1)  将fined query给标记
        memory_mask = memory_mask.permute(0, 2, 1).reshape(memory_mask.shape[0], 1, H, W) # (B_n, 1, H, W)

        # 获取稀疏特征图，训练时使用GT来截取，推理时使用single检测结果截取
        memory_sparse = memory.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W) # (B_n, 256, H, W) 
        if valid_bboxes_single is not None:
            rois_lst = valid_bboxes_single # [N1, N2, N3, N4] 每个场景中每个single的bbx
        memory_sparse = self.get_sparse_features_func(memory_sparse, rois_lst) # (B_n, 256, H, W) 
        # memory_sparse = memory_sparse.flatten(2).permute(0, 2, 1) # (B_n, HW, C) 
        # print("memory_sparse shape is ", memory_sparse.shape)

        memory_batch_lst = self.regroup(memory, record_len)
        memory_discrete_batch_lst = self.regroup(memory_discrete, record_len)
        ref_boxes_before_trans_batch_lst = self.regroup(ref_boxes_before_trans, record_len)
        memory_mask_batch_lst = self.regroup(memory_mask, record_len)

        ego_topk_indexes_batch_lst = self.regroup(ego_topk_indexes, record_len)
        extra_topk_indexes_batch_lst = self.regroup(extra_topk_indexes, record_len)
        extra_query_batch_lst = self.regroup(extra_query, record_len)
        extra_topk_proposals_batch_lst = self.regroup(extra_topk_proposals, record_len) #  [(N1, extra_num, 8), (N2, extra_num, 8)...]

        memory_sparse_batch_lst =  self.regroup(memory_sparse, record_len)

        fused_queries = []
        fused_ref_windows = []
        fused_indicies = []
        ego_features = []
        # 将其他的agent的feature 投影到ego坐标系
        for bid in range(len(memory_batch_lst)):
            N = record_len[bid] # number of valid agent
            
            memory_b = memory_batch_lst[bid] # (N, H*W, C) 单独一个样本下的N个agent，其中第一个为ego的feature
            memory_sparse_b = memory_sparse_batch_lst[bid] # (N, C, H, W) 稀疏特征图
            memory_discrete_b = memory_discrete_batch_lst[bid] # (N, C, H, W) Encoder筛选过的留下来，其余全部为空
            ref_boxes_trans_b = ref_boxes_before_trans_batch_lst[bid][:,:7,:,:] # (N, 7, H, W) Encoder筛选过的留下来，其余全部为空
            ref_probs_trans_b = ref_boxes_before_trans_batch_lst[bid][:,7:,:,:] # (N, 1, H, W) Encoder筛选过的留下来，其余全部为空
            memory_mask_b = memory_mask_batch_lst[bid] # (N, 1, H, W)
            t_matrix = pairwise_t_matrix[bid][:N, :N, :, :] # (N, N, 2, 3)
            t_matrix_ref = pairwise_t_matrix_ref[bid][:N, :N, :, :] # (N, N, 4, 4)
            
            neighbor_memory = warp_affine_simple(memory_discrete_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, C, H, W)
            ref_boxes_trans_b = warp_affine_simple(ref_boxes_trans_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 7, H, W)
            neighbor_memory_mask = warp_affine_simple(memory_mask_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 1, H, W)

            neighbor_memory_sparse_b = warp_affine_simple(memory_sparse_b, t_matrix[0, :, :, :], (H, W), mode='bilinear') # (N, C, H, W)

            # import matplotlib.pyplot as plt
            # import os
            # if self.sample_idx % 20 == 0:
            #     save_dir = "./feature_visualizations"
            #     os.makedirs(save_dir, exist_ok=True)
            #     for b in range(N):
            #         feature_map = neighbor_memory_sparse_b[b]
            #         feature_map = feature_map.mean(dim=0)

            #         # 将特征图归一化到 [0, 255]
            #         def normalize_to_image(tensor):
            #             tensor = tensor - tensor.min()
            #             tensor = tensor / tensor.max()
            #             return (tensor * 255).byte()
                    
            #         dense_feature = normalize_to_image(feature_map)

            #         # 转为 NumPy 格式
            #         dense_feature_np = dense_feature.cpu().numpy()

            #         # 创建可视化画布
            #         plt.figure(figsize=(20, 10))
            #         plt.imshow(dense_feature_np, cmap="viridis")
            #         plt.axis("off")

            #         # 保存到文件
            #         plt.savefig(os.path.join(save_dir, f"trans_feature_map_{self.sample_idx}_{b}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
            #         plt.close() 
            # self.sample_idx += 1

            neighbor_memory_sparse_b = neighbor_memory_sparse_b.flatten(2).permute(0, 2, 1) # (N, HW, C) 
            if memory_b.size(0) != 1: # 注释掉则不使用foreground fusion
                memory_b = torch.cat([memory_b[:1], neighbor_memory_sparse_b[1:]], dim=0)
                memory_b = self.foreground_fusion(memory_b) # (1, H*W, C)

            ref_boxes_trans_b = torch.cat([ref_boxes_trans_b, ref_probs_trans_b], dim=1) # (N, 8, H, W)
            neighbor_memory = neighbor_memory.flatten(2).permute(0, 2, 1) # (N, HW, C)
            ref_boxes_trans_b = ref_boxes_trans_b.flatten(2).permute(0, 2, 1) # (N, HW, 8)
            neighbor_memory_mask = neighbor_memory_mask.flatten(2).permute(0, 2, 1) # (N, HW, 1) 这个里面有0有1, 1的地方就是对应其有效的query，这些query要先在ego feature上做Local Attention
            # pos_b = src_pos[0:N] # (N, HW, C) NOTE 位置编码每个feature在一开始是完全一样的 所以可以直接取需要的个数

            neighbor_mask = neighbor_memory_mask.squeeze(-1).bool() # (N, HW)
            valid_features_lst = [neighbor_memory[i][neighbor_mask[i]].unsqueeze(0) for i in range(N)] # [(1, n1, C), (1, n2, C)...]
            valid_ref_lst = [ref_boxes_trans_b[i][neighbor_mask[i]].unsqueeze(0) for i in range(N)] # [(1, n1, 8), (1, n2, 8)...]
            record_query_num = torch.tensor([v.size(1) for v in valid_ref_lst]) # [n1, n2, ...]
            # valid_pos_lst = [pos_b[i][neighbor_mask[i]] for i in range(N)] # [(n1, C), (n2, C)...]

            none_ego_features_lst = valid_features_lst[1:] # [(1, n2, C), ...]
            none_ego_ref = valid_ref_lst[1:] # [(1, n2, 8), ...]
            # none_ego_pos = valid_pos_lst[1:]

            none_ego_ref_trans_lst = []
            # 旋转参考框，暂时没搞空间变换矩阵的缩放，如果直接缩放空间变换矩阵则不用encode和decode box，但是目前先以这样的方式验证逻辑 TODO 后面要改
            for id, nef in enumerate(none_ego_ref):
                none_ego_bbox_center = self.box_decode_func(nef[..., :7].squeeze(0)) # (n, 7) 反归一化

                none_ego_bbox_corner = box_utils.boxes_to_corners_3d(none_ego_bbox_center, 'lwh') # (n, 8, 3)
                projected_none_ego_bbox_corner = box_utils.project_box3d(none_ego_bbox_corner.float(), t_matrix_ref[0,id+1].float())
                projected_none_ego_bbox_center = box_utils.corners_to_boxes_3d(projected_none_ego_bbox_corner, 'lwh') # (n, 7)
                projected_none_ego_bbox_center = self.box_encode_func(projected_none_ego_bbox_center) # 重新归一化
                projected_none_ego_bbox_center = torch.cat([projected_none_ego_bbox_center, nef[0, :, 7:]], dim=-1) # # (n, 8)
                none_ego_ref_trans_lst.append(projected_none_ego_bbox_center.unsqueeze(0))

                # 还要将变换后的放入到 valid_ref_lst
                valid_ref_lst[id+1] = none_ego_ref_trans_lst[-1]

            if len(none_ego_features_lst) > 0:
                none_ego_features = torch.cat(none_ego_features_lst, dim=1) # (1, n2+n3+..., C)
                none_ego_ref_trans = torch.cat(none_ego_ref_trans_lst, dim=1) # (1, n2+n3+..., 8)
                # none_ego_pos = torch.cat(none_ego_pos, dim=0) # (n2+n3+..., C) # XXX 考虑一下pos是使用ref还是用ego的位置编码， 目前使用ref作为pos编码 所以这个暂时不需要
            
                # TODO 这里仅仅对query做了 Local Attention，但并没有据此去更新旋转过来的参考框 感觉是需要更新的 
                query_adapt = self.trans_adapter(none_ego_features, memory_b[0:1], src_shape, src_start_index, none_ego_ref_trans) # (1, n2+n3+..., C) 其他agent的query在ego feature上进行Local Attention

                query_adapt_lst = self.regroup(query_adapt.squeeze(0), record_query_num[1:]) # [(n2, C), ...]

                query_lst = [q.unsqueeze(0) for q in query_adapt_lst]  # [(1, n2, C), ...]
            else: # 可能的情况: 1. 距离原因导致只有ego一个feature 2. agent投影过来无query
                query_lst = []

            query_lst = valid_features_lst[0:1] + query_lst  # [(1, n1, C), (1, n2, C)...]

            all_indices = [] # [(1, n1, 1), (1, n2, 1), (1, n3, 1)...] 一共N-1 个, 表示场景中的所有有效query的索引 其中ego我们不用
            for i in range(N):
                neighbor_index = torch.nonzero(neighbor_memory_mask[i].squeeze(-1), as_tuple=False) # (n, 1)
                if neighbor_index.size(0) > 0:
                    all_indices.append(neighbor_index.unsqueeze(0))
            all_indices[0] = ego_topk_indexes_batch_lst[bid][0:1] # (N, query_num, 1)中选择出ego的 即(1, query_num, 1)

            ego_feature = memory_b[0:1] # (1, HW, C)

            # 接下来对相同位置的query进行融合，agent提供的额外信息则放置在extra的位置
            if len(all_indices) > 1:
                fused_query, fused_indices = self.fuse_features_by_index(all_indices, query_lst, self.query_fusion, extra_query_batch_lst[bid][0:1], extra_topk_indexes_batch_lst[bid][0:1]) # (1, 300+200, C), (1, 300+200, 1)
                fused_ref, _ = self.fuse_features_by_index(all_indices, valid_ref_lst, self.ref_fusion, extra_topk_proposals_batch_lst[bid][0:1], extra_topk_indexes_batch_lst[bid][0:1]) # (1, 300+200, 8)
                ego_feature = ego_feature.scatter(1, fused_indices.repeat(1, 1, ego_feature.size(-1)), fused_query)
            else: # 如果到这里，可能是: 1.距离过远导致只有一个ego 2.agent投影过来无query
                fused_query = torch.cat([query_lst[0], extra_query_batch_lst[bid][0:1]], dim=1)
                fused_indices = torch.cat([all_indices[0], extra_topk_indexes_batch_lst[bid][0:1]], dim=1)
                fused_ref = torch.cat([valid_ref_lst[0], extra_topk_proposals_batch_lst[bid][0:1]], dim=1)
                # print("crazy, without overlap! ")
                
            fused_queries.append(fused_query)
            fused_indicies.append(fused_indices)
            fused_ref_windows.append(fused_ref)
            ego_features.append(ego_feature)
        fused_queries = torch.cat(fused_queries, dim=0) # (B, query_num+extra_num, C)
        fused_indicies = torch.cat(fused_indicies, dim=0) # (B, query_num+extra_num, 1)
        fused_ref_windows = torch.cat(fused_ref_windows, dim=0) # (B, query_num+extra_num, 8)
        ego_features = torch.cat(ego_features, dim=0) # (B, HW, C)
        # print("fused_indicies first is ", fused_indicies[0].tolist())
        # print("ego_topk_indexes is ", ego_topk_indexes[0].tolist())
        # print("record_query_num is ", record_query_num)
        # if self.debug < 100:
        #     self.debug += 1
        # else:
        #     xxx
        # 加噪声gt，准备一起参与decoder训练去噪
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            ) # (B, pad_size, 8)
            fused_ref_windows = torch.cat(
                (
                    noised_gt_proposals,
                    fused_ref_windows,
                ),
                dim=1,
            ) # (B, pad_size + all_query_num, 8) while: all_query_num == query_num+extra_num
        init_reference_out = fused_ref_windows[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references = self.decoder(
            query_embed, # None
            query_pos, # None
            ego_features, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B, H*W, 256)
            src_shape, # (1, 2)
            src_start_index, # (0,)
            fused_ref_windows, # (B, all_query_num, 8)
            attn_mask,
        ) # (3, B, pad_size + all_query_num, 256) 每一层的输出的query特征， (3， B, pad_size + all_query_num, 7) 每一层的检测结果

        # optional gt forward 对比学习需要用到的动量更新模型用加噪gt来做对比学习的
        if targets is not None:
            batch_size = len(targets) # 这里是协同标签
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [N1, N2, N3, N4] 此为B=4时的各个样本的GT数
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8) # (B, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2) # 得到去噪gt组数 == 3  2指的是每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    ) # 每一组抽取max_gt_num个 (B, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()  # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (3，B, 4*max_gt_num, 256) 与 (3，B, 4*max_gt_num, 8)
                    None,
                    None,
                    ego_features, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B, H*W, 256)
                    src_shape, # (1, 2)
                    src_start_index, # (0,)
                    gt_proposals, # (B, 4*max_gt_num, 8)
                    gt_attn_mask, #（4*max_gt_num，4*max_gt_num）
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            ) # (B, pad_size + all_query_num + 4*max_gt_num, 8) while: all_query_num == query_num+extra_num 输入decoder前的ref window

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (3, B, pad_size + all_query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (3， B, pad_size + all_query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果

        inter_references_out = inter_references
        '''
        从前往后依次返回: Decoder layer每一层的query, 输入Decoder的参考框, Decoder layer每一层的检测结果, Encoder输出的特征图, 初始化的参考框, ego的最高query_num的索引
        TODO Encoder输出的特征图信息会不会不足? 要不要考虑将query融合后的信息放回去 🌟Updated: Done, 先看看性能
        '''
        return hs, init_reference_out, inter_references_out, memory, src_anchors, ego_topk_indexes

    def fuse_features_by_index(self, index_list, feature_list, fusion_func, extra_future, extra_index):
        """
        根据索引对特征进行融合。

        参数:
        - index_list: list of torch.Tensor, 形状为 (1, n, 1) 的索引张量列表，每个表示有效的索引位置。 eg. [(1,300,1), (1,62,1)...]
        - feature_list: list of torch.Tensor, 形状为 (1, n, C) 的特征图张量列表。  eg. [(1,300,C), (1,62,C)...]
        - fusion_func: Callable, 自定义融合函数, 接受输入 (n, k, C)，返回融合后的张量 (n, 1, C),
                    其中 k 表示参与融合的特征数量。
        - extra_future: (1, 200, C), ego自身refine了500个query, 其中300个参与融合, 后200个用于从前到后填充不重叠的其他agent的query 
        - extra_index: (1, 200, 1)

        返回:
        - fused_features: torch.Tensor, 融合后的特征张量, 形状为 (1, ego_query_num + extra_query_num, C)。  eg. (1, 300+200, C)
        """
        # 检查输入合法性
        assert len(index_list) == len(feature_list), "索引列表和特征图列表长度不一致"
        
        # 统一处理索引，获取所有唯一索引
        all_indices = torch.cat([idx.squeeze(0) for idx in index_list], dim=0)  # (sum(n), 1)
        # 相同的索引意味着相同的位置, (n_unique, ) 和逆映射 (sum(n),) 表示每个元素在unique_indices中的位置
        # FIXME 什么情况? 即使设置不用排序，但是最后结果依然排序，想要稳定去重，只能自己写求unique
        # unique_indices, inverse_indices = torch.unique(all_indices, sorted=False, return_inverse=True) 

        seen = set()
        unique_vals = []
        for val in all_indices:
            scalar_val = val.item() # 这里debug了好久，tensor对象是不可哈希的，没搞明白直接导致这里去重失败，还会出现重复，因此必须转为python标量
            if scalar_val not in seen:
                seen.add(scalar_val)
                unique_vals.append(scalar_val)
        unique_indices = torch.tensor(unique_vals).to(all_indices)

        # 构建每个索引对应的特征列表
        feature_map = {idx.item(): [] for idx in unique_indices} # eg. {id: [(1, C), ...]}
        for idx, features in zip(index_list, feature_list):
            for i, ind in enumerate(idx.squeeze(0).squeeze(-1)): # 遍历每个agent的索引
                feature_map[ind.item()].append(features[:, i, :])  # 按索引存入特征 (1, C)

        # 对每个唯一索引进行融合 然后重新放回去 形成{unique_id: [feature]}
        fused_features = []  # 存储融合后的特征
        for idx in unique_indices:
            features_to_fuse = torch.stack(feature_map[idx.item()], dim=1)  # (1, k, C) 同一个空间位置有多个feature, 可能是ego和其他agent，也可能是agent之间
            fused_features.append(fusion_func(features_to_fuse)) # 融合返回的应该是(1, 1, C)
        fused_features = torch.cat(fused_features, dim=1)  # (1, n_unique, C)

        # 从 fused_features 中提取属于 ego 的特征
        ego_indices = index_list[0].squeeze(0).squeeze(-1)  # ego 的索引 （n1,） ego的索引个数是固定的，就等于query_num
        ego_mask = torch.isin(unique_indices, ego_indices)  # 找到属于 ego 的索引 (n_unique, ) ego对应的索引就为 True
        ego_features = fused_features[:, ego_mask, :]  # 提取属于 ego 的部分 (1, ego_query_size, C)

        non_overlap_features = []
        for idx, features in zip(index_list[1:], feature_list[1:]): # 忽略 ego
            mask = ~torch.isin(idx.squeeze(0), index_list[0].squeeze(0)) # 非重叠部分 (n_unique, 1) XXX 首先完全重叠不可能，那只有一种可能，那就是agent和ego感知范围都不重合，所以根本就是空
            selected_features = features[:, mask.squeeze(), :] # 提取非重叠特征 (1, k', C)
            if selected_features.size(1) > 0:
                non_overlap_features.append(selected_features)

        # 将非重叠特征按分数截断并填充到最终结果中
        if len(non_overlap_features) > 0:
            non_overlap_features = torch.cat(non_overlap_features, dim=1)  # (1, k_all, C)
            append_num = min(non_overlap_features.size(1), self.extra_query_num) # 最大不超过 extra_query_num
            extra_future[:, :append_num, :] = non_overlap_features[:,:append_num,:]
        # else: # 首先能进入融合函数就说明有投影的query存在，结果非重叠的特征是0，这就说明全部是重叠的特征, 经过验证，此时投影过来的特征数量很少，一般是个位数，极少数时候是几十
        #     print("------------------------------------------------")
        #     print("Oops! All overlap???")
        #     print("unique_indices shape is ", unique_indices.shape)
        #     print("agent 1 shape is ", index_list[1].shape)
        #     print("------------------------------------------------")

        # 最终特征: ego + extra_future
        final_features = torch.cat([ego_features, extra_future], dim=1)  # (1, ego_query_size + etra_query_num, C)

        unique_indices = unique_indices.unsqueeze(0).unsqueeze(-1) # (1, n_unique, 1)
        index_num = min(unique_indices.size(1), self.num_queries + self.extra_query_num)
        assert unique_indices.size(1) >= self.num_queries
        remain_start = index_num - self.num_queries
        final_indices = torch.cat([unique_indices[:, :index_num, :], extra_index[:, remain_start:, :]], dim = 1) # 500
        return final_features, final_indices

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = Box3dAttention(d_model, nlevel, nhead, with_rotation=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, src_shape, src_start_idx, ref_windows):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            src,
            src_shape,
            None,
            src_start_idx,
            None,
            ref_windows,
        )

        src = src + self.dropout1(src2[0])
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, encoder_layer, num_layers):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)

    def forward(self, src, pos, src_shape, src_start_idx, ref_windows):
        output = src
        for layer in self.layers:
            output = layer(output, pos, src_shape, src_start_idx, ref_windows)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = Box3dAttention(d_model, nlevel, nhead, with_rotation=True)

        self.pos_embed_layer = MLP(8, d_model, d_model, 3)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, idx, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None):
        if idx == 0:
            query = self.pos_embed_layer(ref_windows)
            q = k = query
        elif query_pos is None:
            query_pos = self.pos_embed_layer(ref_windows)
            q = k = self.with_pos_embed(query, query_pos)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = query.transpose(0, 1)

        query2 = self.self_attn(q, k, v, attn_mask=attn_mask)[0]
        query2 = query2.transpose(0, 1)
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(
            self.with_pos_embed(query, query_pos),
            memory,
            memory_shape,
            None,
            memory_start_idx,
            None,
            ref_windows[..., :7],
        )[0]

        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, decoder_layer, num_layers, cp_flag):
        super().__init__()

        self.layers = get_clones(decoder_layer, num_layers)
        self.cp_flag = cp_flag # True
        if self.cp_flag:
            print("===使用checkpoint优化内存, 但是会降低训练速度===")

    def forward(self, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None, return_bboxes=False):
        output = query
        intermediate = []
        intermediate_ref_windows = []
        bboxes_per_layer = []
        for idx, layer in enumerate(self.layers):
            if self.cp_flag:
                output = cp.checkpoint(layer, idx, output, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask)
            else:
                output = layer(idx, output, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask)
            new_ref_logits, new_ref_windows = self.detection_head(output, ref_windows[..., :7], idx)
            # new_ref_logits, new_ref_windows, new_ref_ious = self.detection_head(output, ref_windows[..., :7], idx)
            new_ref_probs = new_ref_logits.sigmoid() # (B, num_queries, 1)
            ref_windows = torch.cat(
                (
                    new_ref_windows.detach(),
                    new_ref_probs.detach(),
                ),
                dim=-1,
            )
            intermediate.append(output)
            intermediate_ref_windows.append(new_ref_windows)
            if return_bboxes:
                self.iou_rectifier = 0.68

                # # quality based select
                # mask = (new_ref_probs > 0.2).int() # (B, num_queries, 1)
                # new_ref_ious = (new_ref_ious + 1) / 2
                # temp_probs = torch.pow(new_ref_probs, 1 - self.iou_rectifier) * torch.pow(new_ref_ious, self.iou_rectifier)
                # new_ref_probs = new_ref_probs * (1 - mask) + mask * temp_probs

                res_boxes = torch.cat(
                    (
                        new_ref_windows.detach(),
                        new_ref_probs.detach(),
                        # new_ref_ious.detach(),
                    ),
                    dim=-1,
                )
                bboxes_per_layer.append(res_boxes) # (Bn, num_queries, 8)
        if return_bboxes:
            return torch.stack(intermediate), torch.stack(intermediate_ref_windows), torch.stack(bboxes_per_layer)
        return torch.stack(intermediate), torch.stack(intermediate_ref_windows)

class TransAdapt(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation):
        super().__init__()
        self.multihead_attn = Box3dAttention(d_model, nlevel, nhead, with_rotation=True)

        self.pos_embed_layer = MLP(8, d_model, d_model, 3)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, memory, memory_shape, memory_start_idx, ref_windows):
        '''
        query: (1, n, C)
        memory: (1, HW, C)
        memory_shape: (1,2)
        memory_start_idx: (0,)
        ref_windows: (1, n, 8)
        '''
        if query.size(1) == 0: # 如果其他agent的query数是0，那就直接return即可
            return query
        query_pos = self.pos_embed_layer(ref_windows)

        query2 = self.multihead_attn(
            self.with_pos_embed(query, query_pos),
            memory,
            memory_shape,
            None,
            memory_start_idx,
            None,
            ref_windows[..., :7],
        )[0]

        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query


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

    mask_flatten = torch.cat(mask_flatten, dim=1) if mask_list is not None else None # (B, L)
    tensor_flatten = torch.cat(tensor_flatten, dim=1)

    return tensor_flatten, mask_flatten, tensor_shape

class TransformerFeature(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
        box_encode_func=None, 
        box_decode_func=None, 
        get_sparse_features_func=None
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom

        self.box_encode_func=box_encode_func
        self.box_decode_func=box_decode_func
        self.get_sparse_features_func=get_sparse_features_func

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        self.fused_encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        self.foreground_fusion = MaxFusion()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)
        self.sample_idx = 0

    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

        out_probs = out_logits[..., 0].sigmoid()
        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False) # 不排序，担心这成为一种先验知识被学到
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)
        # print("out_probs  is ", [round(x, 3) for x in out_probs[0][:1000].tolist()])

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, valid_bboxes_single = None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None, score_mask=None):
        '''
        通信使用稀疏特征图,先做融合再走ConQueR的pipeline, 流程为:
        ⚡️方案一 需要额外建立第二次Encoder的损失 undo
        1️⃣ 所有Feature一起经过Encoder 得到初步ROI区域 2️⃣ 利用ROI区域形成伪图去截取初始的Feature 再利用MaxFusion融合入Ego Feature 这就形成了协同Feature
        3️⃣ 基于协同Feature 重新走 ConQueR
        ⚡️ 方案二 需要监督mask
        1️⃣ 所有的Feature全部预测前景Mask 2️⃣ 利用mask来形成稀疏的Feature并用MAXFusion合入ego 3️⃣ 监督自车
        src: [(B_n, 256, H, W)]
        pos: [(B_n, 256, H, W)]
        noised_gt_box: (B, pad_size, 7)  这里用的应该是协同gt
        noised_gt_onehot: (B, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # 创造参考框，这个是BoxAttention必须的 (B_n, HW, 7)
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B_n, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B_n, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)
        score_mask = score_mask.flatten(-2) # (B_n, H*W)
        
        thresholds = torch.quantile(score_mask, 0.7, dim=1, keepdim=True) # 求70%的分位数
        # thresholds = 0.1
        score_mask = score_mask >= thresholds
        score_mask = score_mask.unsqueeze(-1).expand_as(src)
        src_sparse = src * score_mask # (B_n, H*W, C)

        H, W = src_shape[0,0], src_shape[0,1]
        src_sparse = src_sparse.permute(0, 2, 1).reshape(src.shape[0], src.shape[-1], H, W) # (B_n, C, H, W)

        src_batch_lst = self.regroup(src, record_len)
        src_sparse_batch_lst = self.regroup(src_sparse, record_len)
        fused_features = []
        for b_i in range(len(src_batch_lst)):
            N = record_len[b_i] # number of valid agent
            src_b = src_batch_lst[b_i] # (N, HW, C)
            src_sparse_b = src_sparse_batch_lst[b_i] # (N, C, H, W)
            t_matrix = pairwise_t_matrix[b_i][:N, :N, :, :] # (N, N, 2, 3)
            neighbor_src_sparse_b = warp_affine_simple(src_sparse_b, t_matrix[0, :, :, :], (H, W), mode='bilinear') # (N, C, H, W)
            neighbor_src_sparse_b = neighbor_src_sparse_b.flatten(2).permute(0, 2, 1) # (N, HW, C) 
            early_features = torch.cat([src_b[:1], neighbor_src_sparse_b[1:]], dim=0)
            early_ego_features = self.foreground_fusion(early_features) # (1, H*W, C)
            fused_features_b = torch.cat([early_ego_features[:1], src_b[1:]], dim=0) # TODO 这里有问题 不应该cat 投影后的特征 NOTE 🌟Done
            fused_features.append(fused_features_b)
        fused_features = torch.cat(fused_features, dim=0)
        memory = self.encoder(fused_features, src_pos, src_shape, src_start_index, src_anchors) # BoxAttention 提取特征 结果为(B_n, H*W, 256)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors) # 返回None，None，(B_n, query_num+extra_num, 8)，(B_n, query_num+extra_num, 1)

        memory_batch_lst = self.regroup(memory, record_len)
        topk_proposals_batch_lst = self.regroup(topk_proposals, record_len)
        # memory_batch_lst = self.regroup(memory, record_len)
        ego_memory = []
        ego_topk_proposals = []
        for b_i in range(len(memory_batch_lst)):
            ego_memory.append(memory_batch_lst[b_i][0:1])
            ego_topk_proposals.append(topk_proposals_batch_lst[b_i][0:1])
        ego_memory = torch.cat(ego_memory, dim=0) # (B, HW, C)
        ego_topk_proposals = torch.cat(ego_topk_proposals, dim=0) # (B, query_num， 8)
        
        # H, W = src_shape[0,0], src_shape[0,1]

        # src_sparse = src # (B_n, H*W, 256)
        # if valid_bboxes_single is not None:
        #     rois_lst = valid_bboxes_single # [N1, N2, N3, N4] 每个场景中每个single的bbx
        #     src_raw = src.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W)
        #     src_sparse = self.get_sparse_features_func(src_raw, rois_lst) # (B_n, 256, H, W)
        #     # src_sparse = src_sparse.flatten(2).permute(0, 2, 1)

        # early_feature_lst = self.regroup(src, record_len)
        # src_sparse_lst = self.regroup(src_sparse, record_len)
        # src_anchors_lst = self.regroup(src_anchors, record_len)
        # src_pos_lst = self.regroup(src_pos, record_len)
        # early_feature = []
        # ego_anchors = []
        # ego_pos = []
        # for b_i in range(len(early_feature_lst)):
        #     N = record_len[b_i] # number of valid agent
            

        #     t_matrix = pairwise_t_matrix[b_i][:N, :N, :, :] # (N, N, 2, 3)
        #     early_feature_b = early_feature_lst[b_i] # (N, H*W, 256)
        #     src_sparse_b = src_sparse_lst[b_i] # (N, 256, H, W)
        #     src_anchors_b = src_anchors_lst[b_i]
        #     src_pos_b = src_pos_lst[b_i] # (N, H*W, 256)

        #     neighbor_src_sparse_b = warp_affine_simple(src_sparse_b, t_matrix[0, :, :, :], (H, W), mode='bilinear') # (N, C, H, W)
        #     neighbor_src_sparse_b = neighbor_src_sparse_b.flatten(2).permute(0, 2, 1) # (N, HW, C) 

        #     if early_feature_b.size(0) != 1: # 将其他agent的feature转为稀疏
        #         early_feature_ego_b = torch.cat([early_feature_b[:1], neighbor_src_sparse_b[1:]], dim=0)
        #         early_feature_ego_b = self.foreground_fusion(early_feature_ego_b) # (1, H*W, C)
        #     else:
        #         early_feature_ego_b = early_feature_b

        #     early_feature.append(early_feature_ego_b)
        #     ego_anchors.append(src_anchors_b[0:1])
        #     ego_pos.append(src_pos_b[0:1])
        # early_feature = torch.cat(early_feature, dim=0) # (B,  H*W, 256)  只有ego的feature 融合了来自其他agent的feature
        # ego_anchors = torch.cat(ego_anchors, dim=0) # (B, HW, 7)  只有ego的feature 融合了来自其他agent的feature
        # ego_pos = torch.cat(ego_pos, dim=0) # (B, HW, 256)  只有ego的feature 融合了来自其他agent的feature

        # ego_memory = self.encoder(early_feature, ego_pos, src_shape, src_start_index, ego_anchors) # BoxAttention 提取特征 结果为(B, H*W, 256)
        # query_embed, query_pos, ego_topk_proposals, ego_topk_indexes = self._get_enc_proposals(ego_memory, ego_anchors) # 返回None，None，(B, query_num+extra_num, 8)，(B, query_num+extra_num, 1)
     
        # 加噪声gt，准备一起参与decoder训练去噪
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            ) # (B, pad_size, 8)
            ego_topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    ego_topk_proposals,
                ),
                dim=1,
            ) # (B, pad_size + all_query_num, 8) while: all_query_num == query_num+extra_num
        init_reference_out = ego_topk_proposals[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references = self.decoder(
            query_embed, # None
            query_pos, # None
            ego_memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B, H*W, 256)
            src_shape, # (1, 2)
            src_start_index, # (0,)
            ego_topk_proposals, # (B, all_query_num, 8)
            attn_mask,
        ) # (3, B, pad_size + all_query_num, 256) 每一层的输出的query特征， (3， B, pad_size + all_query_num, 7) 每一层的检测结果

        # optional gt forward 对比学习需要用到的动量更新模型用加噪gt来做对比学习的
        if targets is not None:
            batch_size = len(targets) # 这里是协同标签
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [N1, N2, N3, N4] 此为B=4时的各个样本的GT数
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8) # (B, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2) # 得到去噪gt组数 == 3  2指的是每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    ) # 每一组抽取max_gt_num个 (B, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()  # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (3，B, 4*max_gt_num, 256) 与 (3，B, 4*max_gt_num, 8)
                    None,
                    None,
                    ego_memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B, H*W, 256)
                    src_shape, # (1, 2)
                    src_start_index, # (0,)
                    gt_proposals, # (B, 4*max_gt_num, 8)
                    gt_attn_mask, #（4*max_gt_num，4*max_gt_num）
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            ) # (B, pad_size + all_query_num + 4*max_gt_num, 8) while: all_query_num == query_num+extra_num 输入decoder前的ref window

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (3, B, pad_size + all_query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (3， B, pad_size + all_query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果

        inter_references_out = inter_references
        '''
        从前往后依次返回: Decoder layer每一层的query, 输入Decoder的参考框, Decoder layer每一层的检测结果, Encoder输出的特征图, 初始化的参考框, ego的最高query_num的索引
        TODO Encoder输出的特征图信息会不会不足? 要不要考虑将query融合后的信息放回去 🌟Updated: Done, 先看看性能
        '''
        return hs, init_reference_out, inter_references_out, memory, src_anchors, topk_indexes
    

class TransformerInstance(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
        box_encode_func=None, 
        box_decode_func=None, 
        get_sparse_features_func=None,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom

        self.box_encode_func=box_encode_func
        self.box_decode_func=box_decode_func
        self.get_sparse_features_func=get_sparse_features_func

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        # self.trans_adapter = TransAdapt(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        # self.query_fusion = SimpleGatingFusion()
        # self.ref_fusion = BoxGatingFusion()
        # self.foreground_fusion = MaxFusion()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)
        self.group_atten = GroupAttention(d_model)

        self.agent_embed = nn.Parameter(torch.Tensor(2, d_model))
        self.pos_embed_layer = MLP(8, d_model, d_model, 3)
        self.sample_idx = 0
        self.parameters_fix()

    def parameters_fix(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False


    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

        out_probs = out_logits[..., 0].sigmoid()
        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False)
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)
        # print("out_probs  is ", [round(x, 3) for x in out_probs[0][:1000].tolist()])

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    # def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
    #     """
    #     根据 heatmap 预先筛选 proposals，并从 logits 中选取最终的 queries，返回原始 heatmap 的 HW 索引。

    #     Args:
    #         enc_embed: 编码的嵌入向量，形状为 [B, L, C]
    #         ref_windows: 参考窗口，形状为 [B, L, 4]
    #         indexes: 用于标识某些元素的索引（可选）
    #         heatmap: 热图，形状为 [B, 1, H, W]

    #     Returns:
    #         out_embed: 筛选后的嵌入向量（未设置逻辑，返回 None）
    #         out_pos: 筛选后的位置编码（未设置逻辑，返回 None）
    #         out_ref_windows: 筛选后的参考窗口
    #         hw_indexes: 筛选后的原始 heatmap HW 索引
    #     """
    #     B, L = enc_embed.shape[:2]
    #     H, W = heatmap.shape[-2:]

    #     # 通过 proposal_head 获取预测 logits 和参考窗口
    #     out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

    #     # Step 1: 从 heatmap 中筛选出高概率区域，并保留 HW 索引
    #     heatmap_flat = heatmap.view(B, -1)  # [B, H*W]
    #     top_proposals = heatmap_flat.argsort(dim=-1, descending=True)[..., :self.num_queries * 2]  # 保留 2 倍数量
    #     hw_indexes = top_proposals  # 保存原始 HW 索引 (B, 2*num_queries)

    #     # 利用 HW 索引从 heatmap_flat 提取概率，筛选 logits 和 ref_windows
    #     filtered_logits = torch.gather(out_logits, 1, top_proposals.unsqueeze(-1).expand(-1, -1, out_logits.shape[-1]))
    #     filtered_ref_windows = torch.gather(ref_windows, 1, top_proposals.unsqueeze(-1).expand(-1, -1, ref_windows.shape[-1]))

    #     # Step 2: 在筛选后的 proposals 中，进一步筛选 num_queries 个
    #     out_probs = filtered_logits[..., 0].sigmoid()
    #     topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False) # (B, num_queries)  both shape

    #     # 获取最终的 HW 索引
    #     final_hw_indexes = torch.gather(hw_indexes, 1, indexes)  # 从原始 HW 索引中提取最终的 topk
    #     topk_probs = topk_probs.unsqueeze(-1)  # 增加最后一维

    #     # print("filtered_ref_windows shape is ", filtered_ref_windows.shape)
    #     # print("indexes shape is ", indexes.shape)
    #     # 获取参考窗口的最终内容
    #     out_ref_windows = torch.gather(filtered_ref_windows, 1, indexes.unsqueeze(-1).expand(-1, -1, filtered_ref_windows.shape[-1]))
    #     out_ref_windows = torch.cat(
    #         (
    #             out_ref_windows.detach(),
    #             topk_probs.detach().expand(-1, -1, filtered_logits.shape[-1]),
    #         ),
    #         dim=-1,
    #     )

    #     # 输出的嵌入和位置信息暂时为 None
    #     out_pos = None
    #     out_embed = None

    #     return out_embed, out_pos, out_ref_windows, final_hw_indexes.unsqueeze(-1)


    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None, heatmap=None):
        '''
        ⚡ 先自车检测， 获得高质量query后传输
        src: [(B_n, 256, H, W)]
        pos: [(B_n, 256, H, W)]
        noised_gt_box: (B_n, pad_size, 7)  这里用的应该是single gt 因为这个要先refine单车 形成优质query
        noised_gt_onehot: (B_n, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # 创造参考框，这个是BoxAttention必须的 (B_n, HW, 7)
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B_n, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B_n, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors) # BoxAttention 提取特征 结果为(B_n, H*W, 256)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors, heatmap=heatmap) # 返回None，None，(B_n, query_num, 8)，(B_n, query_num, 1)
        
        pad_size = 0
        # 加噪声gt，准备一起参与decoder训练去噪
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            ) # (B_n, pad_size, 8)
            pad_size = noised_gt_proposals.size(1)
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            ) # (B_n, pad_size + query_num, 8) 
        init_reference_out = topk_proposals[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references, bboxes_per_layer = self.decoder(
            query_embed, # None 
            query_pos, # None
            memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
            src_shape, # (1, 2)
            src_start_index, # (0,)
            topk_proposals, # (B, query_num, 8)
            attn_mask,
            return_bboxes=True
        ) # (3, B_n, pad_size + query_num, 256) 每一层的输出的query特征， (3， B_n, pad_size + all_query_num, 7) 每一层的检测结果 

        # optional gt forward 对比学习需要用到的动量更新模型用加噪gt来做对比学习的
        if targets is not None:
            batch_size = len(targets) # 这里是single 标签
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [N1, N2, N3, N4] 此为B=4时的各个样本的GT数
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8) # (B, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2) # 得到去噪gt组数 == 3  2指的是每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    ) # 每一组抽取max_gt_num个 (B_n, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()  # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (3，B_n, 4*max_gt_num, 256) 与 (3，B_n, 4*max_gt_num, 8)
                    None,
                    None,
                    memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
                    src_shape, # (1, 2)
                    src_start_index, # (0,)
                    gt_proposals, # (B_n, 4*max_gt_num, 8)
                    gt_attn_mask, #（4*max_gt_num，4*max_gt_num）
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            ) # (B_n, pad_size + query_num + 4*max_gt_num, 7)  输入decoder前的ref window

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果

        inter_references_out = inter_references
        '''
        从前往后依次返回: Decoder layer每一层的query, 输入Decoder的参考框, Decoder layer每一层的检测结果, Encoder输出的特征图, 初始化的参考框, ego的最高query_num的索引
        TODO Encoder输出的特征图信息会不会不足? 要不要考虑将query融合后的信息放回去 🌟Updated: Done, 先看看性能
        '''
        result = {
            'hs':hs, # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            'init_reference_out': init_reference_out,  # (B_n, pad_size + query_num + 4*max_gt_num, 8)  输入decoder前的ref window
            'inter_references_out': inter_references_out,  # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果
            'memory': memory, # 包括此项的以下三项都是用来监督encoder时才会用到的
            'src_anchors': src_anchors,
            'topk_indexes': topk_indexes, # (B_n, query_num, 1) 索引
        }

        fined_query = hs[-1, :, pad_size:pad_size+self.num_queries,:] # (B_n, query_num, 256) 最后一层Decoder layer的输出query
        H, W = src_shape[0,0], src_shape[0,1]

        bboxes_per_layer = bboxes_per_layer[-1, :, pad_size:pad_size+self.num_queries, :] # (B_n, query_num, 8)

        memory_discrete = torch.zeros_like(memory) # (B_n, H*W, 256) 

        memory_discrete = memory_discrete.scatter(1, topk_indexes.repeat(1, 1, memory_discrete.size(-1)), fined_query) # (B_n, H*W, 256) 将query放入到一个空的memory中
        memory_discrete = memory_discrete.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W) # (B_n, C, H, W) 形成稀疏的特征图

        # 新建一个默认参考框，然后将decoder最后一次预测的内容填充进去，这个将会在空间变换后作为分组依据
        boxes_before_trans = copy.deepcopy(src_anchors) # (B_n, HW, 7)
        probs_before_trans = torch.zeros(boxes_before_trans.size(0), boxes_before_trans.size(1), 1).to(boxes_before_trans)
        boxes_before_trans = torch.cat([boxes_before_trans, probs_before_trans], dim=-1) # (B_n, HW, 8)
        boxes_before_trans = boxes_before_trans.scatter(1, topk_indexes.repeat(1, 1, boxes_before_trans.size(-1)), bboxes_per_layer) # (B_n, H*W, 8) 将bbox放入到一个空的特征图中
        boxes_before_trans = boxes_before_trans.permute(0, 2, 1).reshape(memory.shape[0], 8, H, W) # (B_n, 8, H, W) 形成稀疏的特征图

        # 创造mask标记fined query
        valid_flag = torch.ones(fined_query.shape[0], fined_query.shape[1], 1).to(fined_query) # (B_n, query_num, 1) 全1
        memory_mask = torch.zeros(memory.shape[0], memory.shape[1], 1).to(memory) # (B_n, HW, 1)
        memory_mask = memory_mask.scatter(1, topk_indexes.repeat(1, 1, memory_mask.size(-1)), valid_flag) # (B_n, HW, 1)  将fined query给标记
        memory_mask = memory_mask.permute(0, 2, 1).reshape(memory_mask.shape[0], 1, H, W) # (B_n, 1, H, W)

        """ # 所有single先卡置信度阈值, 得到筛选后的结果 因此需要返回一个索引 能从query_num中索引出筛选后的query
        # filter_bbox: [(n1,8), (n2,8) ...],  filter_indice: [(n1,), (n2,)...] 筛选对应的索引
        filter_bbox, filter_indice = self.get_bboxes(bboxes_per_layer)

        memory_discrete = []
        valid_flag = torch.ones(1, fined_query.shape[1], 1).to(fined_query) # (1, query_num, 1) 全1
        memory_mask = []
        select_bbox = []
        for bn_i in range(len(memory_discrete)): # 
            memory_discrete_bn_i = torch.zeros(1, memory.shape[-2], memory.shape[-1]).to(memory) # (1, H*W, 256) 
            memory_mask_bn_i = torch.zeros(1, memory.shape[1], 1).to(memory) # (1, HW, 1)
            bbox_bn_i = memory_discrete_bn_i.new_zeros(1, memory.shape[-2], 8) # (1, HW, 8)

            filter_indice_bn_i = filter_indice[bn_i].unsqueeze(-1) # (n, 1) 针对query_num 的索引
            filter_bbox_bn_i = filter_bbox[bn_i].unsqueeze(0) # (1, n, 8)

            select_indexes_bn_i = torch.gather(topk_indexes[bn_i], 0, filter_indice_bn_i.expand(-1, 1)) # 从(query_num, 1)的query中取出筛选出来的那部分 (n, 1) 这就是全局索引了
            select_indexes_bn_i = select_indexes_bn_i.unsqueeze(0) # (1, n, 1)
            fined_query_bn_i = torch.gather(fined_query[bn_i], 0, filter_indice_bn_i.expand(-1, fined_query[bn_i].shape[-1])) # (query_num, 256) 中选出 n, 256

            bbox_bn_i = bbox_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, bbox_bn_i.size(-1)), filter_bbox_bn_i) # 将(1, n, 8) 放入到 （1， HW， 8）
            bbox_bn_i = bbox_bn_i.permute(0, 2, 1).reshape(1, bbox_bn_i.shape[-1], H, W) # (1, 8, H, W) 形成稀疏的特征图

            memory_discrete_bn_i = memory_discrete_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, memory_discrete_bn_i.size(-1)), fined_query_bn_i.unsqueeze(0)) 
            memory_discrete_bn_i = memory_discrete_bn_i.permute(0, 2, 1).reshape(1, memory.shape[-1], H, W) # (1, C, H, W) 形成稀疏的特征图

            memory_mask_bn_i = memory_mask_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, memory_mask_bn_i.size(-1)), valid_flag) # (1, HW, 1)  将fined query给标记
            memory_mask_bn_i = memory_mask_bn_i.permute(0, 2, 1).reshape(memory_mask_bn_i.shape[0], 1, H, W) # (1, 1, H, W)

            select_bbox.append(bbox_bn_i)
            memory_discrete.append(memory_discrete_bn_i)
            memory_mask.append(memory_mask_bn_i) 

        select_bbox = torch.cat(select_bbox, dim=0) # (B_n, 8, H, W) 筛选后的高质量query对应的bbox
        memory_discrete = torch.cat(memory_discrete, dim=0) # (B_n, C, H, W) 筛选后的高质量query已经放入这个memory中
        memory_mask = torch.cat(memory_mask, dim=0) # (B_n, 1, H, W) 被放入的位置标记为1 """

        # 到这里，准备了 1️⃣离散特征图 2️⃣ 离散特征图对应的mask，用来索引和标记 3️⃣ 筛选出来的对应bbox
        memory_discrete_batch_lst = self.regroup(memory_discrete, record_len)
        memory_mask_batch_lst = self.regroup(memory_mask, record_len)
        boxes_before_trans_batch_lst = self.regroup(boxes_before_trans, record_len)

        # memory_batch_lst = self.regroup(memory, record_len)
        all_queries = []
        ref_bboxes = []
        for bid in range(len(record_len)):
            N = record_len[bid] # number of valid agent
            t_matrix = pairwise_t_matrix[bid][:N, :N, :, :] # (N, N, 2, 3)
            t_matrix_ref = pairwise_t_matrix_ref[bid][:N, :N, :, :] # (N, N, 4, 4)
            select_bbox_b = boxes_before_trans_batch_lst[bid] # (N, 8, H，W) 
            memory_discrete_b = memory_discrete_batch_lst[bid] # (N, C, H, W)
            memory_mask_b = memory_mask_batch_lst[bid] # (N, 1, H, W)

            # memory_b = memory_batch_lst[bid] # (N, HW, C)
            # memory_b = memory_b.permute(0, 2, 1).reshape(memory_b.shape[0], memory_b.shape[-1], H, W) 

            # neighbor_memory_dense = warp_affine_simple(memory_b, t_matrix[0, :, :, :], (H, W), mode='bilinear') # (N, C, H, W)


            neighbor_memory = warp_affine_simple(memory_discrete_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, C, H, W)
            neighbor_memory_mask = warp_affine_simple(memory_mask_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 1, H, W)
            neighbor_select_bbox_b = warp_affine_simple(select_bbox_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 8, H，W) 

            # import matplotlib.pyplot as plt
            # import os
            # if self.sample_idx % 20 == 0:
            #     save_dir = "./feature_vis_heatmap"
            #     os.makedirs(save_dir, exist_ok=True)
            #     for b in range(N):
            #         confidence = neighbor_select_bbox_b[b, 7, :, :] # (H, W)
            #         mask = (confidence > 0.25).float()
            #         # mask = mask.unsqueeze(1)
            #         feature_map = neighbor_memory[b]
            #         feature_map = feature_map.mean(dim=0)
            #         feature_mask = neighbor_memory_mask[b]
            #         feature_mask = mask

            #         # 将特征图归一化到 [0, 255]
            #         def normalize_to_image(tensor):
            #             tensor = tensor - tensor.min()
            #             tensor = tensor / tensor.max()
            #             return (tensor * 255).byte()
                    
            #         dense_feature = normalize_to_image(feature_map)
            #         feature_mask = normalize_to_image(feature_mask)
            #         # 转为 NumPy 格式
            #         dense_feature_np = dense_feature.cpu().numpy()
            #         feature_mask_np = feature_mask.cpu().numpy()

            #         # 创建可视化画布
            #         fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            #         axes[0].imshow(dense_feature_np, cmap="viridis")
            #         axes[0].set_title("Dense Feature")
            #         axes[0].axis("off")
            #         axes[1].imshow(feature_mask_np, cmap="viridis")
            #         axes[1].set_title("Sparse Mask")
            #         axes[1].axis("off")

            #         # plt.figure(figsize=(20, 10))
            #         # plt.imshow(dense_feature_np, cmap="viridis")
            #         # plt.axis("off")

            #         # 保存到文件
            #         plt.savefig(os.path.join(save_dir, f"trans_feature_map_{self.sample_idx}_{b}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
            #         plt.close() 
            # self.sample_idx += 1
            
            neighbor_memory = neighbor_memory.flatten(2).permute(0, 2, 1) # (N, HW, C)
            neighbor_memory_mask = neighbor_memory_mask.flatten(2).permute(0, 2, 1) # (N, HW, 1) 这个里面有0有1, 1的地方就是对应其有效的query
            neighbor_select_bbox_b = neighbor_select_bbox_b.flatten(2).permute(0, 2, 1) # (N, HW, 8) 

            neighbor_mask = neighbor_memory_mask.squeeze(-1).bool() # (N, HW)
            valid_query_lst = [neighbor_memory[i][neighbor_mask[i]] for i in range(N)] # [(n1, C), (n2, C)...]
            valid_bbox_lst = [neighbor_select_bbox_b[i][neighbor_mask[i]] for i in range(N)] # [(n1, 8), (n2, 8)...] 
            valid_bbox_norm_lst = [] # [(n1, 8), (n2, 8)...] 

            for id in range(len(valid_bbox_lst)):
                valid_box = valid_bbox_lst[id]
                valid_box_center = self.box_decode_func(valid_box[..., :7]) # (n, 7) 反归一化 变到点云坐标系中的坐标
                valid_box_corner = box_utils.boxes_to_corners_3d(valid_box_center, 'lwh') # (n, 8, 3)
                projected_bbox_corner = box_utils.project_box3d(valid_box_corner.float(), t_matrix_ref[0, id].float())
                projected_bbox_center = box_utils.corners_to_boxes_3d(projected_bbox_corner, 'lwh') # (n, 7)
                projected_bbox_center_norm = self.box_encode_func(projected_bbox_center) # 重新归一化

                # projected_bbox_center = torch.cat([projected_bbox_center, valid_box[:, 7:]], dim=-1) # # (n, 8)
                projected_bbox_center_norm = torch.cat([projected_bbox_center_norm, valid_box[:, 7:]], dim=-1) # # (n, 8)

                # valid_bbox_lst[id] = projected_bbox_center # 到这里后所有的box都统一到ego坐标系了 且所有的box都是真实坐标系，非归一化数值
                valid_bbox_norm_lst.append(projected_bbox_center_norm)

            # neighbor_index = torch.nonzero(neighbor_mask, as_tuple=False) # (N, HW)
                
            # 生成网格索引
            i_indices = torch.arange(H, device=neighbor_mask.device).repeat(W).view(1, -1)  # (1, HW) 每H个元素复制一遍，复制W遍
            j_indices = torch.arange(W, device=neighbor_mask.device).repeat_interleave(H).view(1, -1)  # (1, HW) # 这是每个元素复制H遍
            # 扩展索引以匹配批次大小
            i_indices = i_indices.expand(N, -1)  # (N, HW)
            j_indices = j_indices.expand(N, -1)  # (N, HW)

            # 提取有效位置的索引
            # valid_i = i_indices[neighbor_mask == 1]  
            # valid_j = j_indices[neighbor_mask == 1]  # 所有有效位置的 j 坐标

            query_info_lst = []
            for i in range(len(valid_query_lst)): # 遍历每个agent
                n_q = valid_query_lst[i].size(0)
                agent_queries = valid_query_lst[i] # (n, 8)
                # agent_bboxes = valid_bbox_lst[i] # (n, 8)
                agent_bboxes_norm = valid_bbox_norm_lst[i] # (n,8)
                agent_pos_emb = self.pos_embed_layer(agent_bboxes_norm)
                
                valid_mask  = neighbor_mask[i] # (HW,)
                valid_i = i_indices[i][valid_mask == 1] # 所有有效位置的 i 坐标 (n, )
                valid_j = j_indices[i][valid_mask == 1] # 所有有效位置的 j 坐标
                valid_2d_pos = torch.stack([valid_i, valid_j], dim=-1) # (n, 2)
                # print("torch.sum(valid_mask) is ", torch.sum(valid_mask))
                # print("valid_mask is ", valid_mask)
                # print("valid_2d_pos is ", valid_2d_pos)
                for j in range(n_q): # 遍历每个query
                    query_info = {
                        "agent_id": i,
                        "box_norm": agent_bboxes_norm[j][:7], # （7）
                        "position": agent_bboxes_norm[j][:2], # (2) cx, cy
                        "bbox_size": agent_bboxes_norm[j][3:5], # (2) l, w
                        # "heading": agent_bboxes[j][6:7],
                        "2d_pos": valid_2d_pos[j], # (2,) 2d坐标
                        "confidence": agent_bboxes_norm[j][7:],
                        "pos_emb": agent_pos_emb[j], # 256
                        "feature": agent_queries[j]
                    }
                    query_info_lst.append(query_info)
            clusters = self.build_local_groups_fast(query_info_lst)
            fused_query, norm_bboxes = self.fuse_group_features(query_info_lst, clusters, self.group_atten)

            # queries = [q['feature'].unsqueeze(0) for q in fused_query]

            # queries = torch.cat(queries, dim=0) # n_all, 256
            queries = fused_query # n_all, 256
            # print("queries shape is ", queries.shape)

            # ref_bbox = torch.cat(valid_bbox_norm_lst, dim=0)[..., :7] # n_all, 8
            ref_bbox = norm_bboxes # n_all, 8

            # print("clusters num is ", len(clusters))
            # print("queries.shape is ", queries.shape)
            # print("ref_bbox.shape is ", ref_bbox.shape)

            all_queries.append(queries)
            ref_bboxes.append(ref_bbox)
            

        return result, all_queries, ref_bboxes

   
    def build_local_groups(self, all_queries, dist_thresh=1.0, conf_thresh=0.25):
        """
        基于 3D 距离 和 置信度阈值，对 Query 进行简单聚类，形成多个组 (cluster)。
        all_queries: list of dict, 每个 dict 存储 Query 信息
        dist_thresh: float, 表示中心点距离小于该值则认为是“相似目标”
        conf_thresh: float, 只考虑置信度高于此阈值的 query 进行分组
        
        return: list_of_groups, 每个元素是一个 list，存了若干 Query 的 index
        """
        # 先过滤掉置信度过低的
        valid_indices = [i for i, q in enumerate(all_queries) if q['confidence'] >= conf_thresh]
        valid_queries = [all_queries[i] for i in valid_indices] # 所有符合要求的query
        # print("valid_queries num is", len(valid_queries))
        # 如果没有有效的 query，直接返回空
        if len(valid_queries) == 0:
            return []

        # 保存聚类结果
        clusters = []
        visited = [False] * len(valid_queries)

        for i in range(len(valid_queries)):
            if visited[i]:
                continue
            # BFS/DFS 聚类
            queue = [i]
            visited[i] = True
            cluster = [valid_indices[i]]  # 存原始的 index

            while queue: # 从其中某个节点出发，遍历所有未遍历的节点，
                curr = queue.pop(0)
                # curr_pos = valid_queries[curr]['position']
                curr_box_norm = valid_queries[curr]['box_norm']
                # print("curr_box_norm shape is ", curr_box_norm.shape)
                # 遍历剩余的未访问点
                for j in range(len(valid_queries)):
                    if not visited[j]:
                        # candidate_pos = valid_queries[j]['position']
                        candidate_box_norm = valid_queries[j]['box_norm']
                        # 计算 3D 欧几里得距离 这里可以有很多分组标准
                        # dist = torch.norm(curr_pos - candidate_pos).item()
                        # if dist < dist_thresh:
                        #     visited[j] = True
                        #     queue.append(j)
                        #     cluster.append(valid_indices[j])
                        giou = generalized_box3d_iou(
                        box_cxcyczlwh_to_xyxyxy(curr_box_norm[:6]).unsqueeze(0),
                        box_cxcyczlwh_to_xyxyxy(candidate_box_norm[:6]).unsqueeze(0),
                        ) # (1, 1)
                        if giou[0,0] >  0.2:
                            visited[j] = True
                            queue.append(j)
                            cluster.append(valid_indices[j])

            # 一个完整的组
            clusters.append(cluster)
        
        return clusters

    def build_local_groups_fast(self, all_queries, dist_thresh=1.5, conf_thresh=0.25, iou_thresh=0.2):
        """
        基于 GIoU 阈值对 3D box 做快速聚类，去掉逐对 BFS 的显式循环。
        all_queries: list[dict]，每个元素包含
        {
            'box_norm': (7,)  # 这里只演示到 7 维 [cx, cy, cz, l, w, h, heading]
            'confidence': (1,) # 实际可标量
            ...
        }
        返回值: clusters, 其中每个元素是对应到 all_queries 的原 index 列表。
        """
        # 1) 先过滤置信度
        valid_indices = [i for i, q in enumerate(all_queries) if q['confidence'] >= conf_thresh]
        valid_queries = [all_queries[i] for i in valid_indices]
        # print("valid_queries num is", len(valid_queries))

        M = len(valid_indices)
        if M == 0:
            return []

        """ # 2) 把所有 box_norm 拼成 (M, 7) 的张量
        boxes_7d = torch.stack([q['box_norm'] for q in valid_queries], dim=0)  # (M, 7)

        # 3) 将 (cx,cy,cz,l,w,h,heading) 转为 (x1,y1,z1, x2,y2,z2, ...) 之类能给 GIoU 函数直接用的格式
        #    box_cxcyczlwh_to_xyxyxy(boxes_7d)，输出 (M, 6) 或 (M, 8) ...
        boxes_xyxyxy = box_cxcyczlwh_to_xyxyxy(boxes_7d[:, :6])  # (M, 6)
 
        # 4) 一次性计算 (M,M) GIoU 矩阵
        #    generalized_box3d_iou 需要支持 (M,6) x (M,6) 的批量输入并返回 (M,M)
        iou_matrix = generalized_box3d_iou(boxes_xyxyxy, boxes_xyxyxy)  # (M, M)，每个元素是 giou """
        
        coord_2d = torch.stack([q['2d_pos'] for q in valid_queries]).float() # (M,2)
        dist_2d = torch.cdist(coord_2d, coord_2d)
        # print("coord_2d is ", coord_2d)
        # print("dist_2d is ", dist_2d)
        adj = (dist_2d <= dist_thresh)

        # 5) 根据 iou_thresh 构建邻接矩阵 adj: (M, M)，bool
        # adj = (iou_matrix > iou_thresh)

        # 6) 找所有连通分量：这时候可以用“并查集”来做
        parents = list(range(M)) # [0,1,2,...,M-1]
        
        def find(x):
            if parents[x] != x: # 一直找到起始节点
                parents[x] = find(parents[x])
            return parents[x]

        def union(x, y):
            rx = find(x) # 找parents结点
            ry = find(y)
            if rx != ry:
                parents[ry] = rx

        # 两重循环合并连通分量
        for i in range(M):
            for j in range(i + 1, M):
                if adj[i, j]: # 符合要求的合并
                    union(i, j)

        # 7) 把同一个 parent 的 index 放到同一个 cluster 里
        clusters_dict = defaultdict(list)
        for i in range(M):
            root = find(i)  # 找它的起始节点
            clusters_dict[root].append(valid_indices[i])  # 这里放回原始索引

        # 8) 最后返回一个 list of list
        clusters = list(clusters_dict.values()) # 簇中存着满足要求的认为可能是近似的id
        return clusters


    def fuse_group_features(self, all_queries, clusters, group_attn_module):
        """
        针对每个分组，取出其所有 query 的 feature 做自注意力，更新 feature。
        all_queries: list of dict (所有query)
        clusters: list of list, build_local_groups 的输出
        group_attn_module: nn.Module, 可以是 GroupAttention 实例
        """
        device = next(group_attn_module.parameters()).device

        # max_cluster_size = max(len(cluster) for cluster in clusters) if clusters else 0
        # if max_cluster_size == 0: # 一个簇都没有，几乎不可能
        #     return torch.empty(0, self.d_model).to(device), torch.empty(0, 8).to(device)
        
        # batch_groups = []
        # for cluster in clusters:
        #     if len(cluster) == 1:
        #         idx = cluster[0]
        #         per_feature = all_queries[idx]['feature'] + all_queries[idx]['pos_emb'] + self.agent_embed[all_queries[idx]['agent_id']]
        #         batch_groups.append(per_feature.unsqueeze(0))
        #     else:
        #         feats = [all_queries[idx]['feature'] + all_queries[idx]['pos_emb'] + self.agent_embed[all_queries[idx]['agent_id']] for idx in cluster]
        #         feats = torch.stack(feats, dim=0)  # (K, D)
        #         batch_groups.append(feats.unsqueeze(0))

        # # 填充
        # padded_groups = nn.utils.rnn.pad_sequence(batch_groups, batch_first=True, padding_value=0)  # (B, K_max, D)
        # mask = torch.zeros(padded_groups.size(0), padded_groups.size(1)).to(device)
        # for i, cluster in enumerate(clusters): # 每一簇遍历，设置掩码
        #     mask[i, :len(cluster)] = 1

        # # 注意力
        # fused_feats = group_attn_module(padded_groups)  # (B, K_max, D)
        # fused_feats = fused_feats * mask.unsqueeze(-1)

        # # 移除填充
        # fused_feats = fused_feats[mask.bool()]

        # # 收集 bbox
        # new_all_bboxes = []
        # for cluster in clusters:
        #     for idx in cluster:
        #         new_all_bboxes.append(all_queries[idx]['box_norm'].unsqueeze(0))
        # new_all_bboxes = torch.cat(new_all_bboxes, dim=0).to(device)
        
        # return fused_feats, new_all_bboxes

        new_all_queries = []
        new_all_bboxes = []
        for cluster in clusters:
            if len(cluster) == 1:
                # 只有1个元素，不需要融合，跳过
                idx = cluster[0]
                per_feature = all_queries[idx]['feature'] + all_queries[idx]['pos_emb'] + self.agent_embed[all_queries[idx]['agent_id']]
                new_all_bboxes.append(all_queries[idx]['box_norm'].unsqueeze(0))
                new_all_queries.append(per_feature.unsqueeze(0))
                continue
            
            # 组内所有特征收集
            feats = []
            for idx in cluster:
                per_feature = all_queries[idx]['feature'] + all_queries[idx]['pos_emb'] + self.agent_embed[all_queries[idx]['agent_id']]
                feats.append(per_feature.unsqueeze(0))  # [1, D]
            
            # 拼到维度上: (1, K, D)
            feats = torch.cat(feats, dim=0).unsqueeze(0).to(device)  # B=1, K=len(cluster)

            # 做一次自注意力
            fused_feats = group_attn_module(feats)  # (1, K, D)

            new_all_queries.append(fused_feats.squeeze(0))
            # 写回到 all_queries
            for i, idx in enumerate(cluster):
                # all_queries[idx]['feature'] = fused_feats[0, i, :]
                new_all_bboxes.append(all_queries[idx]['box_norm'].unsqueeze(0))
        new_all_queries = torch.cat(new_all_queries, dim=0).to(device)
        new_all_bboxes = torch.cat(new_all_bboxes, dim=0).to(device)
        # print("new_all_bboxes shape is ", new_all_bboxes.shape)
        # print("new_all_queries shape is ", new_all_queries.shape)
        assert new_all_bboxes.size(0) == new_all_queries.size(0)
        return new_all_queries, new_all_bboxes
        return all_queries


class TransformerInstanceV1(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
        box_encode_func=None, 
        box_decode_func=None, 
        get_sparse_features_func=None,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom

        self.box_encode_func=box_encode_func
        self.box_decode_func=box_decode_func
        self.get_sparse_features_func=get_sparse_features_func

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        # self.trans_adapter = TransAdapt(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        # self.query_fusion = SimpleGatingFusion()
        # self.ref_fusion = BoxGatingFusion()
        # self.foreground_fusion = MaxFusion()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)
        self.fd_atten = Fusion_Decoder(d_model)

        self.agent_embed = nn.Parameter(torch.Tensor(2, d_model))
        self.pos_embed_layer = MLP(8, d_model, d_model, 3)
        self.sample_idx = 0
        self.parameters_fix()

    def parameters_fix(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False


    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

        out_probs = out_logits[..., 0].sigmoid()
        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False)
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)
        # print("out_probs  is ", [round(x, 3) for x in out_probs[0][:1000].tolist()])

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    # def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
    #     """
    #     根据 heatmap 预先筛选 proposals，并从 logits 中选取最终的 queries，返回原始 heatmap 的 HW 索引。

    #     Args:
    #         enc_embed: 编码的嵌入向量，形状为 [B, L, C]
    #         ref_windows: 参考窗口，形状为 [B, L, 4]
    #         indexes: 用于标识某些元素的索引（可选）
    #         heatmap: 热图，形状为 [B, 1, H, W]

    #     Returns:
    #         out_embed: 筛选后的嵌入向量（未设置逻辑，返回 None）
    #         out_pos: 筛选后的位置编码（未设置逻辑，返回 None）
    #         out_ref_windows: 筛选后的参考窗口
    #         hw_indexes: 筛选后的原始 heatmap HW 索引
    #     """
    #     B, L = enc_embed.shape[:2]
    #     H, W = heatmap.shape[-2:]

    #     # 通过 proposal_head 获取预测 logits 和参考窗口
    #     out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

    #     # Step 1: 从 heatmap 中筛选出高概率区域，并保留 HW 索引
    #     heatmap_flat = heatmap.view(B, -1)  # [B, H*W]
    #     top_proposals = heatmap_flat.argsort(dim=-1, descending=True)[..., :self.num_queries * 2]  # 保留 2 倍数量
    #     hw_indexes = top_proposals  # 保存原始 HW 索引 (B, 2*num_queries)

    #     # 利用 HW 索引从 heatmap_flat 提取概率，筛选 logits 和 ref_windows
    #     filtered_logits = torch.gather(out_logits, 1, top_proposals.unsqueeze(-1).expand(-1, -1, out_logits.shape[-1]))
    #     filtered_ref_windows = torch.gather(ref_windows, 1, top_proposals.unsqueeze(-1).expand(-1, -1, ref_windows.shape[-1]))

    #     # Step 2: 在筛选后的 proposals 中，进一步筛选 num_queries 个
    #     out_probs = filtered_logits[..., 0].sigmoid()
    #     topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False) # (B, num_queries)  both shape

    #     # 获取最终的 HW 索引
    #     final_hw_indexes = torch.gather(hw_indexes, 1, indexes)  # 从原始 HW 索引中提取最终的 topk
    #     topk_probs = topk_probs.unsqueeze(-1)  # 增加最后一维

    #     # print("filtered_ref_windows shape is ", filtered_ref_windows.shape)
    #     # print("indexes shape is ", indexes.shape)
    #     # 获取参考窗口的最终内容
    #     out_ref_windows = torch.gather(filtered_ref_windows, 1, indexes.unsqueeze(-1).expand(-1, -1, filtered_ref_windows.shape[-1]))
    #     out_ref_windows = torch.cat(
    #         (
    #             out_ref_windows.detach(),
    #             topk_probs.detach().expand(-1, -1, filtered_logits.shape[-1]),
    #         ),
    #         dim=-1,
    #     )

    #     # 输出的嵌入和位置信息暂时为 None
    #     out_pos = None
    #     out_embed = None

    #     return out_embed, out_pos, out_ref_windows, final_hw_indexes.unsqueeze(-1)


    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None, heatmap=None):
        '''
        ⚡ 先自车检测， 获得高质量query后传输
        src: [(B_n, 256, H, W)]
        pos: [(B_n, 256, H, W)]
        noised_gt_box: (B_n, pad_size, 7)  这里用的应该是single gt 因为这个要先refine单车 形成优质query
        noised_gt_onehot: (B_n, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # 创造参考框，这个是BoxAttention必须的 (B_n, HW, 7)
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B_n, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B_n, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors) # BoxAttention 提取特征 结果为(B_n, H*W, 256)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors, heatmap=heatmap) # 返回None，None，(B_n, query_num, 8)，(B_n, query_num, 1)
        
        pad_size = 0
        # 加噪声gt，准备一起参与decoder训练去噪
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            ) # (B_n, pad_size, 8)
            pad_size = noised_gt_proposals.size(1)
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            ) # (B_n, pad_size + query_num, 8) 
        init_reference_out = topk_proposals[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references, bboxes_per_layer = self.decoder(
            query_embed, # None 
            query_pos, # None
            memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
            src_shape, # (1, 2)
            src_start_index, # (0,)
            topk_proposals, # (B, query_num, 8)
            attn_mask,
            return_bboxes=True
        ) # (3, B_n, pad_size + query_num, 256) 每一层的输出的query特征， (3， B_n, pad_size + all_query_num, 7) 每一层的检测结果 

        # optional gt forward 对比学习需要用到的动量更新模型用加噪gt来做对比学习的
        if targets is not None:
            batch_size = len(targets) # 这里是single 标签
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [N1, N2, N3, N4] 此为B=4时的各个样本的GT数
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8) # (B, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2) # 得到去噪gt组数 == 3  2指的是每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    ) # 每一组抽取max_gt_num个 (B_n, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()  # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (3，B_n, 4*max_gt_num, 256) 与 (3，B_n, 4*max_gt_num, 8)
                    None,
                    None,
                    memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
                    src_shape, # (1, 2)
                    src_start_index, # (0,)
                    gt_proposals, # (B_n, 4*max_gt_num, 8)
                    gt_attn_mask, #（4*max_gt_num，4*max_gt_num）
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            ) # (B_n, pad_size + query_num + 4*max_gt_num, 7)  输入decoder前的ref window

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果

        inter_references_out = inter_references
        '''
        从前往后依次返回: Decoder layer每一层的query, 输入Decoder的参考框, Decoder layer每一层的检测结果, Encoder输出的特征图, 初始化的参考框, ego的最高query_num的索引
        TODO Encoder输出的特征图信息会不会不足? 要不要考虑将query融合后的信息放回去 🌟Updated: Done, 先看看性能
        '''
        result = {
            'hs':hs, # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            'init_reference_out': init_reference_out,  # (B_n, pad_size + query_num + 4*max_gt_num, 8)  输入decoder前的ref window
            'inter_references_out': inter_references_out,  # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果
            'memory': memory, # 包括此项的以下三项都是用来监督encoder时才会用到的
            'src_anchors': src_anchors,
            'topk_indexes': topk_indexes, # (B_n, query_num, 1) 索引
        }

        fined_query = hs[-1, :, pad_size:pad_size+self.num_queries,:] # (B_n, query_num, 256) 最后一层Decoder layer的输出query
        H, W = src_shape[0,0], src_shape[0,1]

        bboxes_per_layer = bboxes_per_layer[-1, :, pad_size:pad_size+self.num_queries, :] # (B_n, query_num, 8)

        memory_discrete = torch.zeros_like(memory) # (B_n, H*W, 256) 

        memory_discrete = memory_discrete.scatter(1, topk_indexes.repeat(1, 1, memory_discrete.size(-1)), fined_query) # (B_n, H*W, 256) 将query放入到一个空的memory中
        memory_discrete = memory_discrete.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W) # (B_n, C, H, W) 形成稀疏的特征图

        # 新建一个默认参考框，然后将decoder最后一次预测的内容填充进去，这个将会在空间变换后作为分组依据
        boxes_before_trans = copy.deepcopy(src_anchors) # (B_n, HW, 7)
        probs_before_trans = torch.zeros(boxes_before_trans.size(0), boxes_before_trans.size(1), 1).to(boxes_before_trans)
        boxes_before_trans = torch.cat([boxes_before_trans, probs_before_trans], dim=-1) # (B_n, HW, 8)
        boxes_before_trans = boxes_before_trans.scatter(1, topk_indexes.repeat(1, 1, boxes_before_trans.size(-1)), bboxes_per_layer) # (B_n, H*W, 8) 将bbox放入到一个空的特征图中
        boxes_before_trans = boxes_before_trans.permute(0, 2, 1).reshape(memory.shape[0], 8, H, W) # (B_n, 8, H, W) 形成稀疏的特征图

        # 创造mask标记fined query
        valid_flag = torch.ones(fined_query.shape[0], fined_query.shape[1], 1).to(fined_query) # (B_n, query_num, 1) 全1
        memory_mask = torch.zeros(memory.shape[0], memory.shape[1], 1).to(memory) # (B_n, HW, 1)
        memory_mask = memory_mask.scatter(1, topk_indexes.repeat(1, 1, memory_mask.size(-1)), valid_flag) # (B_n, HW, 1)  将fined query给标记
        memory_mask = memory_mask.permute(0, 2, 1).reshape(memory_mask.shape[0], 1, H, W) # (B_n, 1, H, W)

        """ # 所有single先卡置信度阈值, 得到筛选后的结果 因此需要返回一个索引 能从query_num中索引出筛选后的query
        # filter_bbox: [(n1,8), (n2,8) ...],  filter_indice: [(n1,), (n2,)...] 筛选对应的索引
        filter_bbox, filter_indice = self.get_bboxes(bboxes_per_layer)

        memory_discrete = []
        valid_flag = torch.ones(1, fined_query.shape[1], 1).to(fined_query) # (1, query_num, 1) 全1
        memory_mask = []
        select_bbox = []
        for bn_i in range(len(memory_discrete)): # 
            memory_discrete_bn_i = torch.zeros(1, memory.shape[-2], memory.shape[-1]).to(memory) # (1, H*W, 256) 
            memory_mask_bn_i = torch.zeros(1, memory.shape[1], 1).to(memory) # (1, HW, 1)
            bbox_bn_i = memory_discrete_bn_i.new_zeros(1, memory.shape[-2], 8) # (1, HW, 8)

            filter_indice_bn_i = filter_indice[bn_i].unsqueeze(-1) # (n, 1) 针对query_num 的索引
            filter_bbox_bn_i = filter_bbox[bn_i].unsqueeze(0) # (1, n, 8)

            select_indexes_bn_i = torch.gather(topk_indexes[bn_i], 0, filter_indice_bn_i.expand(-1, 1)) # 从(query_num, 1)的query中取出筛选出来的那部分 (n, 1) 这就是全局索引了
            select_indexes_bn_i = select_indexes_bn_i.unsqueeze(0) # (1, n, 1)
            fined_query_bn_i = torch.gather(fined_query[bn_i], 0, filter_indice_bn_i.expand(-1, fined_query[bn_i].shape[-1])) # (query_num, 256) 中选出 n, 256

            bbox_bn_i = bbox_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, bbox_bn_i.size(-1)), filter_bbox_bn_i) # 将(1, n, 8) 放入到 （1， HW， 8）
            bbox_bn_i = bbox_bn_i.permute(0, 2, 1).reshape(1, bbox_bn_i.shape[-1], H, W) # (1, 8, H, W) 形成稀疏的特征图

            memory_discrete_bn_i = memory_discrete_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, memory_discrete_bn_i.size(-1)), fined_query_bn_i.unsqueeze(0)) 
            memory_discrete_bn_i = memory_discrete_bn_i.permute(0, 2, 1).reshape(1, memory.shape[-1], H, W) # (1, C, H, W) 形成稀疏的特征图

            memory_mask_bn_i = memory_mask_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, memory_mask_bn_i.size(-1)), valid_flag) # (1, HW, 1)  将fined query给标记
            memory_mask_bn_i = memory_mask_bn_i.permute(0, 2, 1).reshape(memory_mask_bn_i.shape[0], 1, H, W) # (1, 1, H, W)

            select_bbox.append(bbox_bn_i)
            memory_discrete.append(memory_discrete_bn_i)
            memory_mask.append(memory_mask_bn_i) 

        select_bbox = torch.cat(select_bbox, dim=0) # (B_n, 8, H, W) 筛选后的高质量query对应的bbox
        memory_discrete = torch.cat(memory_discrete, dim=0) # (B_n, C, H, W) 筛选后的高质量query已经放入这个memory中
        memory_mask = torch.cat(memory_mask, dim=0) # (B_n, 1, H, W) 被放入的位置标记为1 """

        # 到这里，准备了 1️⃣离散特征图 2️⃣ 离散特征图对应的mask，用来索引和标记 3️⃣ 筛选出来的对应bbox
        memory_discrete_batch_lst = self.regroup(memory_discrete, record_len)
        memory_mask_batch_lst = self.regroup(memory_mask, record_len)
        boxes_before_trans_batch_lst = self.regroup(boxes_before_trans, record_len)

        # memory_batch_lst = self.regroup(memory, record_len)
        all_queries = []
        ref_bboxes = []
        solo_bboxes = []

        # print("record_len is ", record_len)
        # print("memory_discrete shape is ", memory_discrete.shape)
        # print("pairwise_t_matrix shape is ", pairwise_t_matrix.shape)
        com_num_batch = []
        for bid in range(len(record_len)):
            N = record_len[bid] # number of valid agent
            t_matrix = pairwise_t_matrix[bid][:N, :N, :, :] # (N, N, 2, 3)
            t_matrix_ref = pairwise_t_matrix_ref[bid][:N, :N, :, :] # (N, N, 4, 4)
            select_bbox_b = boxes_before_trans_batch_lst[bid] # (N, 8, H，W) 
            memory_discrete_b = memory_discrete_batch_lst[bid] # (N, C, H, W)
            memory_mask_b = memory_mask_batch_lst[bid] # (N, 1, H, W)

            # memory_b = memory_batch_lst[bid] # (N, HW, C)
            # memory_b = memory_b.permute(0, 2, 1).reshape(memory_b.shape[0], memory_b.shape[-1], H, W) 

            # neighbor_memory_dense = warp_affine_simple(memory_b, t_matrix[0, :, :, :], (H, W), mode='bilinear') # (N, C, H, W)

            # print("---------------------------------------------------------")
            # print("memory shape is ", memory.shape)
            # print("record_len is ", record_len)
            # print("N is ", N)
            # print("t_matrix shape is ", t_matrix.shape)
            # print("memory_discrete_b shape is ", memory_discrete_b.shape)
            # print("---------------------------------------------------------")

            neighbor_memory = warp_affine_simple(memory_discrete_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, C, H, W)
            neighbor_memory_mask = warp_affine_simple(memory_mask_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 1, H, W)
            neighbor_select_bbox_b = warp_affine_simple(select_bbox_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 8, H，W) 

            """ import matplotlib.pyplot as plt
            # from matplotlib.cm import ScalarMappable
            from sklearn.preprocessing import MinMaxScaler
            from scipy.ndimage import gaussian_filter1d
            from matplotlib.colors import LogNorm, SymLogNorm
            import os
            if self.sample_idx % 20 == 0:
                save_dir = "./feature_vis_final"
                os.makedirs(save_dir, exist_ok=True)
                # 先收集所有feature_map的全局最大最小值
                feature_maps = []
                for b in range(N):
                    feature_map = neighbor_memory[b].mean(dim=0)  # (H, W)
                    feature_maps.append(feature_map)
                # 计算全局最小值和最大值
                global_min = min([fm.min() for fm in feature_maps])
                global_max = max([fm.max() for fm in feature_maps])
                for b in range(N): # 遍历一个场景中的所有agent
                    confidence = neighbor_select_bbox_b[b, 7, :, :] # (H, W)
                    mask = (confidence > 0.1).float()
                    # mask = mask.unsqueeze(1)
                    feature_map = neighbor_memory[b] # (C, H, W)
                    feature_map = feature_map.mean(dim=0) # (H, W)
                    feature_map[feature_map<0]=0
                    feature_mask = neighbor_memory_mask[b] # (1, H, W)
                    feature_mask = mask

                    feature_filtered = copy.deepcopy(feature_map)
                    feature_filtered[feature_mask != 1] = feature_filtered.min()
                    # 将特征图归一化到 [0, 255]
                    def normalize_to_image(tensor):
                        tensor = tensor - tensor.min()
                        tensor = tensor / tensor.max()
                        return (tensor * 255).byte()
                    
                    dense_feature = normalize_to_image(feature_map)
                    feature_mask = normalize_to_image(feature_mask)
                    feature_filtered = normalize_to_image(feature_filtered)
                    # 转为 NumPy 格式
                    dense_feature_np = dense_feature.cpu().numpy()
                    feature_mask_np = feature_mask.cpu().numpy()
                    feature_filtered_np = feature_filtered.cpu().numpy()

                    # 创建可视化画布
                    plt.figure(figsize=(10, 10))  # 调整尺寸为单个图
                    plt.imshow(dense_feature_np, cmap="viridis")
                    plt.title("Dense Feature")
                    plt.axis("off")
                    plt.savefig(os.path.join(save_dir, f"trans_feature_map_{self.sample_idx}_{b}_dense.png"), dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()  # 关闭当前图

                    plt.figure(figsize=(10, 10))  # 调整尺寸为单个图
                    plt.imshow(feature_filtered_np, cmap="viridis")
                    plt.title("Dense Feature")
                    plt.axis("off")
                    plt.savefig(os.path.join(save_dir, f"trans_feature_map_{self.sample_idx}_{b}_filtered.png"), dpi=300, bbox_inches="tight", pad_inches=0)
                    plt.close()  # 关闭当前图
                    # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                    # axes[0].imshow(dense_feature_np, cmap="viridis")
                    # axes[0].set_title("Dense Feature")
                    # axes[0].axis("off")
                    # axes[1].imshow(feature_mask_np, cmap="viridis")
                    # axes[1].set_title("Sparse Mask")
                    # axes[1].axis("off")

                    # # 保存到文件
                    # plt.savefig(os.path.join(save_dir, f"trans_feature_map_{self.sample_idx}_{b}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
                    # plt.close() 
            # self.sample_idx += 1 """
            
            neighbor_memory = neighbor_memory.flatten(2).permute(0, 2, 1) # (N, HW, C)
            neighbor_memory_mask = neighbor_memory_mask.flatten(2).permute(0, 2, 1) # (N, HW, 1) 这个里面有0有1, 1的地方就是对应其有效的query
            neighbor_select_bbox_b = neighbor_select_bbox_b.flatten(2).permute(0, 2, 1) # (N, HW, 8) 

            neighbor_mask = neighbor_memory_mask.squeeze(-1).bool() # (N, HW)
            valid_query_lst = [neighbor_memory[i][neighbor_mask[i]] for i in range(N)] # [(n1, C), (n2, C)...]
            valid_bbox_lst = [neighbor_select_bbox_b[i][neighbor_mask[i]] for i in range(N)] # [(n1, 8), (n2, 8)...]
            # valid_query_lst = valid_query_lst[1:]
            # valid_bbox_lst = valid_bbox_lst[1:]
            valid_bbox_norm_lst = [] # [(n1, 8), (n2, 8)...] 

            for id in range(len(valid_bbox_lst)):
                valid_box = valid_bbox_lst[id] # (n, 8)
                valid_box_center = self.box_decode_func(valid_box[..., :7]) # (n, 7) 反归一化 变到点云坐标系中的坐标
                valid_box_corner = box_utils.boxes_to_corners_3d(valid_box_center, 'lwh') # (n, 8, 3)
                projected_bbox_corner = box_utils.project_box3d(valid_box_corner.float(), t_matrix_ref[id, 0].float()) # TODO WHY???
                projected_bbox_center = box_utils.corners_to_boxes_3d(projected_bbox_corner, 'lwh') # (n, 7)
                projected_bbox_center_norm = self.box_encode_func(projected_bbox_center) # 重新归一化
                # print("t_matrix_ref[0, id] is ", t_matrix_ref[0, id])
                # print("t_matrix_ref[id, 0] is ", t_matrix_ref[id, 0])
                # xxx
                # projected_bbox_center = torch.cat([projected_bbox_center, valid_box[:, 7:]], dim=-1) # # (n, 8)
                projected_bbox_center_norm = torch.cat([projected_bbox_center_norm, valid_box[:, 7:]], dim=-1) # # (n, 8)
                # projected_bbox_center_norm = valid_box 

                # valid_bbox_lst[id] = projected_bbox_center # 到这里后所有的box都统一到ego坐标系了 且所有的box都是真实坐标系，非归一化数值
                valid_bbox_norm_lst.append(projected_bbox_center_norm)

            # neighbor_index = torch.nonzero(neighbor_mask, as_tuple=False) # (N, HW)
                
            # 生成网格索引
            i_indices = torch.arange(H, device=neighbor_mask.device).repeat(W).view(1, -1)  # (1, HW) 每H个元素复制一遍，复制W遍
            j_indices = torch.arange(W, device=neighbor_mask.device).repeat_interleave(H).view(1, -1)  # (1, HW) # 这是每个元素复制H遍
            # 扩展索引以匹配批次大小
            i_indices = i_indices.expand(N, -1)  # (N, HW)
            j_indices = j_indices.expand(N, -1)  # (N, HW)

            # 提取有效位置的索引
            # valid_i = i_indices[neighbor_mask == 1]  
            # valid_j = j_indices[neighbor_mask == 1]  # 所有有效位置的 j 坐标

            query_info_lst = []
            com_num = 0
            for i in range(len(valid_query_lst)): # 遍历每个agent
                n_q = valid_query_lst[i].size(0)
                agent_queries = valid_query_lst[i] # (n, 8)
                # agent_bboxes = valid_bbox_lst[i] # (n, 8)
                agent_bboxes_norm = valid_bbox_norm_lst[i] # (n,8)
                agent_pos_emb = self.pos_embed_layer(agent_bboxes_norm)
                
                valid_mask  = neighbor_mask[i] # (HW,)
                valid_i = i_indices[i][valid_mask == 1] # 所有有效位置的 i 坐标 (n, )
                valid_j = j_indices[i][valid_mask == 1] # 所有有效位置的 j 坐标
                valid_2d_pos = torch.stack([valid_i, valid_j], dim=-1) # (n, 2)
                # print("torch.sum(valid_mask) is ", torch.sum(valid_mask))
                # print("valid_mask is ", valid_mask)
                # print("valid_2d_pos is ", valid_2d_pos)
                for j in range(n_q): # 遍历每个query
                    query_info = {
                        "agent_id": i,
                        "box_norm": agent_bboxes_norm[j][:7], # （7）
                        "position": agent_bboxes_norm[j][:2], # (2) cx, cy
                        "bbox_size": agent_bboxes_norm[j][3:5], # (2) l, w
                        # "heading": agent_bboxes[j][6:7],
                        "2d_pos": valid_2d_pos[j], # (2,) 2d坐标
                        "confidence": agent_bboxes_norm[j][7:],
                        "pos_emb": agent_pos_emb[j], # 256
                        "feature": agent_queries[j]
                    }
                    # if i > 0: # 消融实验
                    #     if agent_bboxes_norm[j][7:] > 0.1:
                    #         query_info_lst.append(query_info)
                    # else:
                    query_info_lst.append(query_info)

                    # 计算通信量 也就是传输的query 个数
                    if i > 0:
                        if agent_bboxes_norm[j][7:] > 0.1:
                            com_num += 1
            extra_agents_num = max(1, N-1)
            com_num /= extra_agents_num
            com_num_batch.append(com_num)

            # 🌟 我们的主张是 将所有的query对应的box放在一起，判断两两iou，如果有某个和其他所有box的重合度都为0.1或者以下，认为它是独立检测，则这个不需要交互，直接参与最后的匹配
            attn_mask, valid_indicies, indep_queries = gaussian_atten_mask_from_bboxes(query_info_lst, decode_box_func=self.box_decode_func) # (M, M)的Mask
            # attn_mask = None
            if attn_mask is not None:
                valid_feat = []
                valid_feat_pos = []
                norm_bboxes = []
                for vid in valid_indicies:
                    per_query_feat = query_info_lst[vid]['feature']
                    # per_query_feat_w_pos = query_info_lst[vid]['feature'] + query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_pos = query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_box = query_info_lst[vid]['box_norm']

                    valid_feat.append(per_query_feat.unsqueeze(0)) # (1, D)
                    valid_feat_pos.append(per_query_pos.unsqueeze(0)) # (1, D)
                    norm_bboxes.append(per_query_box.unsqueeze(0)) # (1, 7)
                valid_feat = torch.cat(valid_feat, dim=0).unsqueeze(0) # (1, M, D)
                valid_feat_pos = torch.cat(valid_feat_pos, dim=0).unsqueeze(0) # (1, M, D)
                norm_bboxes = torch.cat(norm_bboxes, dim=0) # (M, 7)

                """ coop_query_num = valid_feat.shape[1]
                if self.sample_idx % 20 == 0:
                    scaler = MinMaxScaler()
                    query_data = valid_feat.squeeze(0).cpu().numpy()  # shape: (coop_query_num, 256)
                    # 全局统计量
                    global_min = query_data.min()
                    global_max = query_data.max()
                    q25, q75 = np.percentile(query_data, [25, 75])
                    # 动态范围设置（排除离群值）
                    iqr = q75 - q25
                    vmin = max(global_min, q25 - 1.5*iqr)
                    vmax = min(global_max, q75 + 1.5*iqr)
                    
                    # 平滑处理
                    sigma = 1.5
                    smoothed = gaussian_filter1d(query_data, sigma=sigma, axis=1)
                    
                    # 创建画布
                    fig, axes = plt.subplots(coop_query_num, 1, 
                                            figsize=(15, 2*coop_query_num),
                                            gridspec_kw={'hspace':0.4})
                    # 使用锐化的颜色映射
                    cmap = plt.cm.nipy_spectral
                    norm = plt.Normalize(vmin, vmax)
                    
                    for i, ax in enumerate(axes):
                        # 绘制增强对比度的条形
                        colors = cmap(norm(smoothed[i]))
                        ax.bar(range(256), [1]*256, 
                            width=1.0, 
                            color=colors,
                            edgecolor='none')
                        
                        # 添加维度标记
                        if i == coop_query_num-1:
                            ax.set_xticks([0, 64, 128, 192, 255])
                            ax.set_xticklabels(['0', '64', '128', '192', '255'])
                        else:
                            ax.set_xticks([])
                            
                        ax.set_yticks([])
                        ax.set_title(f'Query {i+1} (Min:{smoothed[i].min():.2f}, Max:{smoothed[i].max():.2f})', 
                                    fontsize=8, pad=2)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                    # 添加全局颜色条
                    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    fig.colorbar(sm, cax=cax, label='Global Normalized Value')
                    
                    plt.savefig(os.path.join(save_dir, f'query_vis_{self.sample_idx}_{b}before.png'), 
                            dpi=300, bbox_inches='tight')
                    plt.close()

                    vis_mask = attn_mask.cpu().numpy()

                    raw_vmin = np.min(vis_mask[np.isfinite(vis_mask)])

                    raw_vmax = np.max(vis_mask[np.isfinite(vis_mask)])
                    # 处理极端值（如-inf）
                    vis_mask = np.nan_to_num(vis_mask, nan=0.0, posinf=raw_vmax, neginf=raw_vmin)
                    # 创建画布
                    plt.figure(figsize=(10, 10), dpi=300)

                    # 计算有效数值范围
                    vmin = np.min(vis_mask[np.isfinite(vis_mask)])
                    vmax = np.max(vis_mask[np.isfinite(vis_mask)])

                    # 处理全零或单一值情况
                    if vmin == vmax:
                        vmin -= 1e-6
                        vmax += 1e-6

                    # 判断是否使用符号敏感归一化
                    is_signed = vmin < 0

                    # 动态选择归一化方式
                    if is_signed:
                        # 计算线性阈值（至少1e-5防止除零）
                        linthresh = max(0.1 * max(abs(vmin), abs(vmax)), 1e-5)
                        norm = SymLogNorm(linthresh=linthresh, 
                                        linscale=0.5,
                                        vmin=vmin,  # 使用实际范围
                                        vmax=vmax)
                        cmap = 'RdBu_r'
                    else:
                        # 确保LogNorm最小值合法
                        safe_vmin = max(vmin, 1e-6) if vmin <= 0 else vmin
                        norm = LogNorm(vmin=safe_vmin, vmax=vmax)
                        cmap = 'viridis'

                    # 绘制主图
                    im = plt.imshow(vis_mask, 
                                cmap=cmap,
                                norm=norm,
                                interpolation='nearest',
                                origin='upper')
                    
                    # 安全添加颜色条
                    if norm.vmin < norm.vmax:  # 最终验证范围合法性
                        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                        cbar.set_label('Attention Weight (Log Scale)')
                    else:
                        plt.text(0.5, 0.5, "Invalid Value Range", 
                                ha='center', va='center', 
                                transform=plt.gca().transAxes)
                    
                    # 设置坐标轴
                    # plt.xlabel("Key Positions")
                    # plt.ylabel("Query Positions")
                    # plt.title(f"{title}\nValue Range: [{vmin:.2f}, {vmax:.2f}]")
                    
                    # 添加辅助等高线（可选）
                    # if n <= 20:
                    # 绘制等高线帮助识别数值变化
                    # levels = np.linspace(vmin, vmax, 8)
                    # cs = plt.contour(vis_mask, levels=levels, 
                    #                 colors='black', linewidths=0.5)
                    # plt.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'atten_{self.sample_idx}_{b}.png'), dpi=300, bbox_inches='tight')
                    plt.close() """

                fused_query = self.fd_atten(valid_feat, valid_feat_pos, attn_mask)
                # fused_query = valid_feat # 消融实验

                """ if self.sample_idx % 20 == 0:
                    scaler = MinMaxScaler()
                    query_data = fused_query.squeeze(0).cpu().numpy()  # shape: (coop_query_num, 256)
                    # 全局统计量
                    global_min = query_data.min()
                    global_max = query_data.max()
                    q25, q75 = np.percentile(query_data, [25, 75])
                    # 动态范围设置（排除离群值）
                    iqr = q75 - q25
                    vmin = max(global_min, q25 - 1.5*iqr)
                    vmax = min(global_max, q75 + 1.5*iqr)
                    
                    # 平滑处理
                    sigma = 1.5
                    smoothed = gaussian_filter1d(query_data, sigma=sigma, axis=1)
                    
                    # 创建画布
                    fig, axes = plt.subplots(coop_query_num, 1, 
                                            figsize=(15, 2*coop_query_num),
                                            gridspec_kw={'hspace':0.4})
                    # 使用锐化的颜色映射
                    cmap = plt.cm.nipy_spectral
                    norm = plt.Normalize(vmin, vmax)
                    
                    for i, ax in enumerate(axes):
                        # 绘制增强对比度的条形
                        colors = cmap(norm(smoothed[i]))
                        ax.bar(range(256), [1]*256, 
                            width=1.0, 
                            color=colors,
                            edgecolor='none')
                        
                        # 添加维度标记
                        if i == coop_query_num-1:
                            ax.set_xticks([0, 64, 128, 192, 255])
                            ax.set_xticklabels(['0', '64', '128', '192', '255'])
                        else:
                            ax.set_xticks([])
                            
                        ax.set_yticks([])
                        ax.set_title(f'Query {i+1} (Min:{smoothed[i].min():.2f}, Max:{smoothed[i].max():.2f})', 
                                    fontsize=8, pad=2)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)

                    # 添加全局颜色条
                    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    fig.colorbar(sm, cax=cax, label='Global Normalized Value')
                    
                    plt.savefig(os.path.join(save_dir, f'query_vis_{self.sample_idx}_{b}after.png'), 
                            dpi=300, bbox_inches='tight')
                    plt.close()

                    # # queries_scaled = scaler.fit_transform(fused_query.squeeze(0).cpu())
                    # # 将二维数据展开为一维进行全局归一化
                    # global_scaled = scaler.fit_transform(fused_query.squeeze(0).cpu().reshape(-1, 1)).reshape(coop_query_num, 256)
                    # # queries_scaled = gaussian_filter1d(global_scaled, sigma=1.5, axis=1)
                    # sigma = 1.5
                    # queries_scaled = gaussian_filter1d(global_scaled, sigma=sigma, axis=1)
                    # fig, axes = plt.subplots(coop_query_num, 1, figsize=(15, 2*coop_query_num))
                    # for i, ax in enumerate(axes):
                    #     # 将每个query的256维展开为颜色条
                    #     cmap = plt.cm.viridis  # 选择颜色映射
                    #     norm = plt.Normalize(vmin=0, vmax=1)
                    #     bars = ax.bar(range(256), [1]*256,  # 高度统一为1
                    #                 width=1.0, 
                    #                 color=cmap(queries_scaled[i]))
                        
                    #     ax.set_xticks([])
                    #     ax.set_yticks([])
                    #     ax.set_title(f'Query {i+1} Vector Visualization', pad=10)
                    #     ax.spines['top'].set_visible(False)
                    #     ax.spines['right'].set_visible(False)
                    #     ax.spines['bottom'].set_visible(False)
                    #     ax.spines['left'].set_visible(False)
                    # # 添加公共颜色条
                    # # cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    # # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    # # fig.colorbar(sm, cax=cax, label='Normalized Value')

                    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # 留出颜色条位置
                    # plt.savefig(os.path.join(save_dir, f'query_vector_bars_{self.sample_idx}_{b}_later.png'), dpi=300, bbox_inches='tight')
                    # plt.close() """


                queries = fused_query.squeeze(0) # n_all, 256
                # print("queries shape is ", queries.shape)

                # ref_bbox = torch.cat(valid_bbox_norm_lst, dim=0)[..., :7] # n_all, 8
                ref_bbox = norm_bboxes # n_all, 7
            else:
                queries = ref_bbox = None

            if len(indep_queries) > 0:
                indep_boxes = []
                for indep_query in indep_queries:
                    indep_box = torch.cat((indep_query['box_norm'], indep_query['confidence']), dim=-1) # (8)
                    indep_boxes.append(indep_box.unsqueeze(0))
                indep_boxes = torch.cat(indep_boxes, dim=0) # (indep_num, 8)
                indep_boxes = indep_boxes.unsqueeze(0) # (1, indep_num, 8)
            else:
                indep_boxes = None
                indep_boxes = torch.zeros(1, 1, 8).to(memory)

            all_queries.append(queries)
            ref_bboxes.append(ref_bbox)
            solo_bboxes.append(indep_boxes)
            
            self.sample_idx += 1

        return result, all_queries, ref_bboxes, solo_bboxes, com_num_batch

class TransformerInstanceV2(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
        box_encode_func=None, 
        box_decode_func=None, 
        get_sparse_features_func=None,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom

        self.box_encode_func=box_encode_func
        self.box_decode_func=box_decode_func
        self.get_sparse_features_func=get_sparse_features_func

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        # self.trans_adapter = TransAdapt(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        # self.query_fusion = SimpleGatingFusion()
        # self.ref_fusion = BoxGatingFusion()
        # self.foreground_fusion = MaxFusion()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)
        self.fd_atten = Fusion_Decoder(d_model)

        self.agent_embed = nn.Parameter(torch.Tensor(5, d_model))
        self.pos_embed_layer = MLP(8, d_model, d_model, 3)
        self.sample_idx = 0
        self.iou_rectifier = 0.68

        self.parameters_fix()

    def parameters_fix(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False


    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows, out_ious = self.proposal_head(enc_embed, ref_windows) # 分类logits(B, H * W, 1)  、boxes 定位 (B, H * W, 7) 、 IoU预测 (B, H * W, 1)

        out_probs = out_logits[..., 0].sigmoid() # (B, HW)

        mask = (out_probs > 0.2).int() # (B, H * W, )  大于阈值的全部置为True

        out_ious = (out_ious + 1) / 2
        if isinstance(self.iou_rectifier, float): # 大于阈值的部分需要同时考虑IoU得分
            temp_probs = torch.pow(out_probs, 1 - self.iou_rectifier) * torch.pow(out_ious[..., 0], self.iou_rectifier)
            out_probs = out_probs * (1 - mask) + mask * temp_probs
        else:
            raise TypeError('only list or float')

        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False)
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    # def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
    #     """
    #     根据 heatmap 预先筛选 proposals，并从 logits 中选取最终的 queries，返回原始 heatmap 的 HW 索引。

    #     Args:
    #         enc_embed: 编码的嵌入向量，形状为 [B, L, C]
    #         ref_windows: 参考窗口，形状为 [B, L, 4]
    #         indexes: 用于标识某些元素的索引（可选）
    #         heatmap: 热图，形状为 [B, 1, H, W]

    #     Returns:
    #         out_embed: 筛选后的嵌入向量（未设置逻辑，返回 None）
    #         out_pos: 筛选后的位置编码（未设置逻辑，返回 None）
    #         out_ref_windows: 筛选后的参考窗口
    #         hw_indexes: 筛选后的原始 heatmap HW 索引
    #     """
    #     B, L = enc_embed.shape[:2]
    #     H, W = heatmap.shape[-2:]

    #     # 通过 proposal_head 获取预测 logits 和参考窗口
    #     out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

    #     # Step 1: 从 heatmap 中筛选出高概率区域，并保留 HW 索引
    #     heatmap_flat = heatmap.view(B, -1)  # [B, H*W]
    #     top_proposals = heatmap_flat.argsort(dim=-1, descending=True)[..., :self.num_queries * 2]  # 保留 2 倍数量
    #     hw_indexes = top_proposals  # 保存原始 HW 索引 (B, 2*num_queries)

    #     # 利用 HW 索引从 heatmap_flat 提取概率，筛选 logits 和 ref_windows
    #     filtered_logits = torch.gather(out_logits, 1, top_proposals.unsqueeze(-1).expand(-1, -1, out_logits.shape[-1]))
    #     filtered_ref_windows = torch.gather(ref_windows, 1, top_proposals.unsqueeze(-1).expand(-1, -1, ref_windows.shape[-1]))

    #     # Step 2: 在筛选后的 proposals 中，进一步筛选 num_queries 个
    #     out_probs = filtered_logits[..., 0].sigmoid()
    #     topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False) # (B, num_queries)  both shape

    #     # 获取最终的 HW 索引
    #     final_hw_indexes = torch.gather(hw_indexes, 1, indexes)  # 从原始 HW 索引中提取最终的 topk
    #     topk_probs = topk_probs.unsqueeze(-1)  # 增加最后一维

    #     # print("filtered_ref_windows shape is ", filtered_ref_windows.shape)
    #     # print("indexes shape is ", indexes.shape)
    #     # 获取参考窗口的最终内容
    #     out_ref_windows = torch.gather(filtered_ref_windows, 1, indexes.unsqueeze(-1).expand(-1, -1, filtered_ref_windows.shape[-1]))
    #     out_ref_windows = torch.cat(
    #         (
    #             out_ref_windows.detach(),
    #             topk_probs.detach().expand(-1, -1, filtered_logits.shape[-1]),
    #         ),
    #         dim=-1,
    #     )

    #     # 输出的嵌入和位置信息暂时为 None
    #     out_pos = None
    #     out_embed = None

    #     return out_embed, out_pos, out_ref_windows, final_hw_indexes.unsqueeze(-1)


    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None, heatmap=None):
        '''
        ⚡ 先自车检测， 获得高质量query后传输
        src: [(B_n, 256, H, W)]
        pos: [(B_n, 256, H, W)]
        noised_gt_box: (B_n, pad_size, 7)  这里用的应该是single gt 因为这个要先refine单车 形成优质query
        noised_gt_onehot: (B_n, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # 创造参考框，这个是BoxAttention必须的 (B_n, HW, 7)
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B_n, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B_n, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors) # BoxAttention 提取特征 结果为(B_n, H*W, 256)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors, heatmap=heatmap) # 返回None，None，(B_n, query_num, 8)，(B_n, query_num, 1)
        
        pad_size = 0
        # 加噪声gt，准备一起参与decoder训练去噪
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            ) # (B_n, pad_size, 8)
            pad_size = noised_gt_proposals.size(1)
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            ) # (B_n, pad_size + query_num, 8) 
        init_reference_out = topk_proposals[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references, bboxes_per_layer = self.decoder(
            query_embed, # None 
            query_pos, # None
            memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
            src_shape, # (1, 2)
            src_start_index, # (0,)
            topk_proposals, # (B, query_num, 8)
            attn_mask,
            return_bboxes=True
        ) # (3, B_n, pad_size + query_num, 256) 每一层的输出的query特征， (3， B_n, pad_size + all_query_num, 7) 每一层的检测结果 

        # optional gt forward 对比学习需要用到的动量更新模型用加噪gt来做对比学习的
        if targets is not None:
            batch_size = len(targets) # 这里是single 标签
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [N1, N2, N3, N4] 此为B=4时的各个样本的GT数
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8) # (B, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2) # 得到去噪gt组数 == 3  2指的是每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    ) # 每一组抽取max_gt_num个 (B_n, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()  # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (3，B_n, 4*max_gt_num, 256) 与 (3，B_n, 4*max_gt_num, 8)
                    None,
                    None,
                    memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
                    src_shape, # (1, 2)
                    src_start_index, # (0,)
                    gt_proposals, # (B_n, 4*max_gt_num, 8)
                    gt_attn_mask, #（4*max_gt_num，4*max_gt_num）
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            ) # (B_n, pad_size + query_num + 4*max_gt_num, 7)  输入decoder前的ref window

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果

        inter_references_out = inter_references
        '''
        从前往后依次返回: Decoder layer每一层的query, 输入Decoder的参考框, Decoder layer每一层的检测结果, Encoder输出的特征图, 初始化的参考框, ego的最高query_num的索引
        TODO Encoder输出的特征图信息会不会不足? 要不要考虑将query融合后的信息放回去 🌟Updated: Done, 先看看性能
        '''
        result = {
            'hs':hs, # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            'init_reference_out': init_reference_out,  # (B_n, pad_size + query_num + 4*max_gt_num, 8)  输入decoder前的ref window
            'inter_references_out': inter_references_out,  # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果
            'memory': memory, # 包括此项的以下三项都是用来监督encoder时才会用到的
            'src_anchors': src_anchors,
            'topk_indexes': topk_indexes, # (B_n, query_num, 1) 索引
        }

        fined_query = hs[-1, :, pad_size:pad_size+self.num_queries,:] # (B_n, query_num, 256) 最后一层Decoder layer的输出query
        H, W = src_shape[0,0], src_shape[0,1]

        bboxes_per_layer = bboxes_per_layer[-1, :, pad_size:pad_size+self.num_queries, :] # (B_n, query_num, 8)

        memory_discrete = torch.zeros_like(memory) # (B_n, H*W, 256) 

        memory_discrete = memory_discrete.scatter(1, topk_indexes.repeat(1, 1, memory_discrete.size(-1)), fined_query) # (B_n, H*W, 256) 将query放入到一个空的memory中
        memory_discrete = memory_discrete.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W) # (B_n, C, H, W) 形成稀疏的特征图

        # 新建一个默认参考框，然后将decoder最后一次预测的内容填充进去，这个将会在空间变换后作为分组依据
        boxes_before_trans = copy.deepcopy(src_anchors) # (B_n, HW, 7)
        probs_before_trans = torch.zeros(boxes_before_trans.size(0), boxes_before_trans.size(1), 2).to(boxes_before_trans)
        boxes_before_trans = torch.cat([boxes_before_trans, probs_before_trans], dim=-1) # (B_n, HW, 9)
        boxes_before_trans = boxes_before_trans.scatter(1, topk_indexes.repeat(1, 1, boxes_before_trans.size(-1)), bboxes_per_layer) # (B_n, H*W, 8) 将bbox放入到一个空的特征图中
        boxes_before_trans = boxes_before_trans.permute(0, 2, 1).reshape(memory.shape[0], 9, H, W) # (B_n, 9, H, W) 形成稀疏的特征图

        # 创造mask标记fined query
        valid_flag = torch.ones(fined_query.shape[0], fined_query.shape[1], 1).to(fined_query) # (B_n, query_num, 1) 全1
        memory_mask = torch.zeros(memory.shape[0], memory.shape[1], 1).to(memory) # (B_n, HW, 1)
        memory_mask = memory_mask.scatter(1, topk_indexes.repeat(1, 1, memory_mask.size(-1)), valid_flag) # (B_n, HW, 1)  将fined query给标记
        memory_mask = memory_mask.permute(0, 2, 1).reshape(memory_mask.shape[0], 1, H, W) # (B_n, 1, H, W)

        """ # 所有single先卡置信度阈值, 得到筛选后的结果 因此需要返回一个索引 能从query_num中索引出筛选后的query
        # filter_bbox: [(n1,8), (n2,8) ...],  filter_indice: [(n1,), (n2,)...] 筛选对应的索引
        filter_bbox, filter_indice = self.get_bboxes(bboxes_per_layer)

        memory_discrete = []
        valid_flag = torch.ones(1, fined_query.shape[1], 1).to(fined_query) # (1, query_num, 1) 全1
        memory_mask = []
        select_bbox = []
        for bn_i in range(len(memory_discrete)): # 
            memory_discrete_bn_i = torch.zeros(1, memory.shape[-2], memory.shape[-1]).to(memory) # (1, H*W, 256) 
            memory_mask_bn_i = torch.zeros(1, memory.shape[1], 1).to(memory) # (1, HW, 1)
            bbox_bn_i = memory_discrete_bn_i.new_zeros(1, memory.shape[-2], 8) # (1, HW, 8)

            filter_indice_bn_i = filter_indice[bn_i].unsqueeze(-1) # (n, 1) 针对query_num 的索引
            filter_bbox_bn_i = filter_bbox[bn_i].unsqueeze(0) # (1, n, 8)

            select_indexes_bn_i = torch.gather(topk_indexes[bn_i], 0, filter_indice_bn_i.expand(-1, 1)) # 从(query_num, 1)的query中取出筛选出来的那部分 (n, 1) 这就是全局索引了
            select_indexes_bn_i = select_indexes_bn_i.unsqueeze(0) # (1, n, 1)
            fined_query_bn_i = torch.gather(fined_query[bn_i], 0, filter_indice_bn_i.expand(-1, fined_query[bn_i].shape[-1])) # (query_num, 256) 中选出 n, 256

            bbox_bn_i = bbox_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, bbox_bn_i.size(-1)), filter_bbox_bn_i) # 将(1, n, 8) 放入到 （1， HW， 8）
            bbox_bn_i = bbox_bn_i.permute(0, 2, 1).reshape(1, bbox_bn_i.shape[-1], H, W) # (1, 8, H, W) 形成稀疏的特征图

            memory_discrete_bn_i = memory_discrete_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, memory_discrete_bn_i.size(-1)), fined_query_bn_i.unsqueeze(0)) 
            memory_discrete_bn_i = memory_discrete_bn_i.permute(0, 2, 1).reshape(1, memory.shape[-1], H, W) # (1, C, H, W) 形成稀疏的特征图

            memory_mask_bn_i = memory_mask_bn_i.scatter(1, select_indexes_bn_i.repeat(1, 1, memory_mask_bn_i.size(-1)), valid_flag) # (1, HW, 1)  将fined query给标记
            memory_mask_bn_i = memory_mask_bn_i.permute(0, 2, 1).reshape(memory_mask_bn_i.shape[0], 1, H, W) # (1, 1, H, W)

            select_bbox.append(bbox_bn_i)
            memory_discrete.append(memory_discrete_bn_i)
            memory_mask.append(memory_mask_bn_i) 

        select_bbox = torch.cat(select_bbox, dim=0) # (B_n, 8, H, W) 筛选后的高质量query对应的bbox
        memory_discrete = torch.cat(memory_discrete, dim=0) # (B_n, C, H, W) 筛选后的高质量query已经放入这个memory中
        memory_mask = torch.cat(memory_mask, dim=0) # (B_n, 1, H, W) 被放入的位置标记为1 """

        # 到这里，准备了 1️⃣离散特征图 2️⃣ 离散特征图对应的mask，用来索引和标记 3️⃣ 筛选出来的对应bbox
        memory_discrete_batch_lst = self.regroup(memory_discrete, record_len)
        memory_mask_batch_lst = self.regroup(memory_mask, record_len)
        boxes_before_trans_batch_lst = self.regroup(boxes_before_trans, record_len)

        # memory_batch_lst = self.regroup(memory, record_len)
        all_queries = []
        ref_bboxes = []
        solo_bboxes = []
        for bid in range(len(record_len)):
            N = record_len[bid] # number of valid agent
            t_matrix = pairwise_t_matrix[bid][:N, :N, :, :] # (N, N, 2, 3)
            t_matrix_ref = pairwise_t_matrix_ref[bid][:N, :N, :, :] # (N, N, 4, 4)
            select_bbox_b = boxes_before_trans_batch_lst[bid] # (N, 8, H，W) 
            memory_discrete_b = memory_discrete_batch_lst[bid] # (N, C, H, W)
            memory_mask_b = memory_mask_batch_lst[bid] # (N, 1, H, W)

            # memory_b = memory_batch_lst[bid] # (N, HW, C)
            # memory_b = memory_b.permute(0, 2, 1).reshape(memory_b.shape[0], memory_b.shape[-1], H, W) 

            # neighbor_memory_dense = warp_affine_simple(memory_b, t_matrix[0, :, :, :], (H, W), mode='bilinear') # (N, C, H, W)


            neighbor_memory = warp_affine_simple(memory_discrete_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, C, H, W)
            neighbor_memory_mask = warp_affine_simple(memory_mask_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 1, H, W)
            neighbor_select_bbox_b = warp_affine_simple(select_bbox_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 8, H，W) 

            # import matplotlib.pyplot as plt
            # import os
            # if self.sample_idx % 20 == 0:
            #     save_dir = "./feature_vis_gaussian"
            #     os.makedirs(save_dir, exist_ok=True)
            #     for b in range(N):
            #         confidence = neighbor_select_bbox_b[b, 7, :, :] # (H, W)
            #         mask = (confidence > 0.1).float()
            #         # mask = mask.unsqueeze(1)
            #         feature_map = neighbor_memory[b]
            #         feature_map = feature_map.mean(dim=0)
            #         feature_mask = neighbor_memory_mask[b]
            #         feature_mask = mask

            #         # 将特征图归一化到 [0, 255]
            #         def normalize_to_image(tensor):
            #             tensor = tensor - tensor.min()
            #             tensor = tensor / tensor.max()
            #             return (tensor * 255).byte()
                    
            #         dense_feature = normalize_to_image(feature_map)
            #         feature_mask = normalize_to_image(feature_mask)
            #         # 转为 NumPy 格式
            #         dense_feature_np = dense_feature.cpu().numpy()
            #         feature_mask_np = feature_mask.cpu().numpy()

            #         # 创建可视化画布
            #         fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            #         axes[0].imshow(dense_feature_np, cmap="viridis")
            #         axes[0].set_title("Dense Feature")
            #         axes[0].axis("off")
            #         axes[1].imshow(feature_mask_np, cmap="viridis")
            #         axes[1].set_title("Sparse Mask")
            #         axes[1].axis("off")

            #         # plt.figure(figsize=(20, 10))
            #         # plt.imshow(dense_feature_np, cmap="viridis")
            #         # plt.axis("off")

            #         # 保存到文件
            #         plt.savefig(os.path.join(save_dir, f"trans_feature_map_{self.sample_idx}_{b}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
            #         plt.close() 
            # self.sample_idx += 1
            
            neighbor_memory = neighbor_memory.flatten(2).permute(0, 2, 1) # (N, HW, C)
            neighbor_memory_mask = neighbor_memory_mask.flatten(2).permute(0, 2, 1) # (N, HW, 1) 这个里面有0有1, 1的地方就是对应其有效的query
            neighbor_select_bbox_b = neighbor_select_bbox_b.flatten(2).permute(0, 2, 1) # (N, HW, 8) 

            neighbor_mask = neighbor_memory_mask.squeeze(-1).bool() # (N, HW)
            valid_query_lst = [neighbor_memory[i][neighbor_mask[i]] for i in range(N)] # [(n1, C), (n2, C)...]
            valid_bbox_lst = [neighbor_select_bbox_b[i][neighbor_mask[i]] for i in range(N)] # [(n1, 8), (n2, 8)...]
            # valid_query_lst = valid_query_lst[1:]
            # valid_bbox_lst = valid_bbox_lst[1:]
            valid_bbox_norm_lst = [] # [(n1, 8), (n2, 8)...] 

            for id in range(len(valid_bbox_lst)):
                valid_box = valid_bbox_lst[id] # (n, 8)
                valid_box_center = self.box_decode_func(valid_box[..., :7]) # (n, 7) 反归一化 变到点云坐标系中的坐标
                valid_box_corner = box_utils.boxes_to_corners_3d(valid_box_center, 'lwh') # (n, 8, 3)
                projected_bbox_corner = box_utils.project_box3d(valid_box_corner.float(), t_matrix_ref[id, 0].float())
                projected_bbox_center = box_utils.corners_to_boxes_3d(projected_bbox_corner, 'lwh') # (n, 7)
                projected_bbox_center_norm = self.box_encode_func(projected_bbox_center) # 重新归一化
                # print("t_matrix_ref[0, id] is ", t_matrix_ref[0, id])
                # print("t_matrix_ref[id, 0] is ", t_matrix_ref[id, 0])
                # xxx
                # projected_bbox_center = torch.cat([projected_bbox_center, valid_box[:, 7:]], dim=-1) # # (n, 8)
                projected_bbox_center_norm = torch.cat([projected_bbox_center_norm, valid_box[:, 7:]], dim=-1) # # (n, 8)
                # projected_bbox_center_norm = valid_box 

                # valid_bbox_lst[id] = projected_bbox_center # 到这里后所有的box都统一到ego坐标系了 且所有的box都是真实坐标系，非归一化数值
                valid_bbox_norm_lst.append(projected_bbox_center_norm)

            # neighbor_index = torch.nonzero(neighbor_mask, as_tuple=False) # (N, HW)
                
            # 生成网格索引
            i_indices = torch.arange(H, device=neighbor_mask.device).repeat(W).view(1, -1)  # (1, HW) 每H个元素复制一遍，复制W遍
            j_indices = torch.arange(W, device=neighbor_mask.device).repeat_interleave(H).view(1, -1)  # (1, HW) # 这是每个元素复制H遍
            # 扩展索引以匹配批次大小
            i_indices = i_indices.expand(N, -1)  # (N, HW)
            j_indices = j_indices.expand(N, -1)  # (N, HW)

            # 提取有效位置的索引
            # valid_i = i_indices[neighbor_mask == 1]  
            # valid_j = j_indices[neighbor_mask == 1]  # 所有有效位置的 j 坐标

            query_info_lst = []
            for i in range(len(valid_query_lst)): # 遍历每个agent
                n_q = valid_query_lst[i].size(0)
                agent_queries = valid_query_lst[i] # (n, 8)
                # agent_bboxes = valid_bbox_lst[i] # (n, 8)
                agent_bboxes_norm = valid_bbox_norm_lst[i] # (n,9)
                agent_pos_emb = self.pos_embed_layer(agent_bboxes_norm[..., :8])
                
                valid_mask  = neighbor_mask[i] # (HW,)
                valid_i = i_indices[i][valid_mask == 1] # 所有有效位置的 i 坐标 (n, )
                valid_j = j_indices[i][valid_mask == 1] # 所有有效位置的 j 坐标
                valid_2d_pos = torch.stack([valid_i, valid_j], dim=-1) # (n, 2)
                # print("torch.sum(valid_mask) is ", torch.sum(valid_mask))
                # print("valid_mask is ", valid_mask)
                # print("valid_2d_pos is ", valid_2d_pos)
                for j in range(n_q): # 遍历每个query
                    query_info = {
                        "agent_id": i,
                        "box_norm": agent_bboxes_norm[j][:7], # （7）
                        "position": agent_bboxes_norm[j][:2], # (2) cx, cy
                        "bbox_size": agent_bboxes_norm[j][3:5], # (2) l, w
                        # "heading": agent_bboxes[j][6:7],
                        "2d_pos": valid_2d_pos[j], # (2,) 2d坐标
                        "confidence": agent_bboxes_norm[j][7:8],
                        "iou_score": agent_bboxes_norm[j][8:],
                        "pos_emb": agent_pos_emb[j], # 256
                        "feature": agent_queries[j]
                    }
                    query_info_lst.append(query_info)
            # 🌟 我们的主张是 将所有的query对应的box放在一起，判断两两iou，如果有某个和其他所有box的重合度都为0.1或者以下，认为它是独立检测，则这个不需要交互，直接参与最后的匹配
            attn_mask, valid_indicies, indep_queries = gaussian_atten_mask_from_bboxes(query_info_lst, decode_box_func=self.box_decode_func) # (M, M)的Mask
            # attn_mask = None
            if attn_mask is not None:
                valid_feat = []
                valid_feat_pos = []
                norm_bboxes = []
                for vid in valid_indicies:
                    per_query_feat = query_info_lst[vid]['feature']
                    # per_query_feat_w_pos = query_info_lst[vid]['feature'] + query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_pos = query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_box = query_info_lst[vid]['box_norm']

                    valid_feat.append(per_query_feat.unsqueeze(0)) # (1, D)
                    valid_feat_pos.append(per_query_pos.unsqueeze(0)) # (1, D)
                    norm_bboxes.append(per_query_box.unsqueeze(0)) # (1, 7)
                valid_feat = torch.cat(valid_feat, dim=0).unsqueeze(0) # (1, M, D)
                valid_feat_pos = torch.cat(valid_feat_pos, dim=0).unsqueeze(0) # (1, M, D)
                norm_bboxes = torch.cat(norm_bboxes, dim=0) # (M, 7)

                fused_query = self.fd_atten(valid_feat, valid_feat_pos, attn_mask)
                # fused_query = valid_feat

                queries = fused_query.squeeze(0) # n_all, 256
                # print("queries shape is ", queries.shape)

                # ref_bbox = torch.cat(valid_bbox_norm_lst, dim=0)[..., :7] # n_all, 8
                ref_bbox = norm_bboxes # n_all, 7
            else:
                queries = ref_bbox = None

            if len(indep_queries) > 0: # 没有意味着所有的query都最起码有一个重叠
                indep_boxes = []
                for indep_query in indep_queries:
                    indep_box = torch.cat((indep_query['box_norm'], indep_query['confidence'], indep_query['iou_score']), dim=-1) # (9)
                    indep_boxes.append(indep_box.unsqueeze(0))
                indep_boxes = torch.cat(indep_boxes, dim=0) # (indep_num, 9)
                indep_boxes = indep_boxes.unsqueeze(0) # (1, indep_num, 9)
            else:
                indep_boxes = None
                indep_boxes = torch.zeros(1, 1, 9).to(memory)

            all_queries.append(queries)
            ref_bboxes.append(ref_bbox)
            solo_bboxes.append(indep_boxes)
            

        return result, all_queries, ref_bboxes, solo_bboxes


class TransformerInstanceV3(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
        box_encode_func=None, 
        box_decode_func=None, 
        get_sparse_features_func=None,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom

        self.box_encode_func=box_encode_func
        self.box_decode_func=box_decode_func
        self.get_sparse_features_func=get_sparse_features_func

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)
        self.fd_atten = Fusion_Decoder(d_model)

        self.agent_embed = nn.Parameter(torch.Tensor(2, d_model))
        self.pos_embed_layer = MLP(8, d_model, d_model, 3)
        self.sample_idx = 0
        self.parameters_fix()

    def parameters_fix(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False


    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

        out_probs = out_logits[..., 0].sigmoid()
        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False)
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)
        # print("out_probs  is ", [round(x, 3) for x in out_probs[0][:1000].tolist()])

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None, heatmap=None):
        '''
        ⚡ 先自车检测， 获得高质量query后传输
        src: [(B_n, 256, H, W)]
        pos: [(B_n, 256, H, W)]
        noised_gt_box: (B_n, pad_size, 7)  这里用的应该是single gt 因为这个要先refine单车 形成优质query
        noised_gt_onehot: (B_n, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # 创造参考框，这个是BoxAttention必须的 (B_n, HW, 7)
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B_n, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B_n, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors) # BoxAttention 提取特征 结果为(B_n, H*W, 256)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors, heatmap=heatmap) # 返回None，None，(B_n, query_num, 8)，(B_n, query_num, 1)
        
        pad_size = 0
        # 加噪声gt，准备一起参与decoder训练去噪
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            ) # (B_n, pad_size, 8)
            pad_size = noised_gt_proposals.size(1)
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            ) # (B_n, pad_size + query_num, 8) 
        init_reference_out = topk_proposals[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references, bboxes_per_layer = self.decoder(
            query_embed, # None 
            query_pos, # None
            memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
            src_shape, # (1, 2)
            src_start_index, # (0,)
            topk_proposals, # (B, query_num, 8)
            attn_mask,
            return_bboxes=True
        ) # (3, B_n, pad_size + query_num, 256) 每一层的输出的query特征， (3， B_n, pad_size + all_query_num, 7) 每一层的检测结果 

        # optional gt forward 对比学习需要用到的动量更新模型用加噪gt来做对比学习的
        if targets is not None:
            batch_size = len(targets) # 这里是single 标签
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [N1, N2, N3, N4] 此为B=4时的各个样本的GT数
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8) # (B, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2) # 得到去噪gt组数 == 3  2指的是每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    ) # 每一组抽取max_gt_num个 (B_n, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()  # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (3，B_n, 4*max_gt_num, 256) 与 (3，B_n, 4*max_gt_num, 8)
                    None,
                    None,
                    memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
                    src_shape, # (1, 2)
                    src_start_index, # (0,)
                    gt_proposals, # (B_n, 4*max_gt_num, 8)
                    gt_attn_mask, #（4*max_gt_num，4*max_gt_num）
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            ) # (B_n, pad_size + query_num + 4*max_gt_num, 7)  输入decoder前的ref window

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果

        inter_references_out = inter_references
        '''
        从前往后依次返回: Decoder layer每一层的query, 输入Decoder的参考框, Decoder layer每一层的检测结果, Encoder输出的特征图, 初始化的参考框, ego的最高query_num的索引
        TODO Encoder输出的特征图信息会不会不足? 要不要考虑将query融合后的信息放回去 🌟Updated: Done, 先看看性能
        '''
        result = {
            'hs':hs, # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            'init_reference_out': init_reference_out,  # (B_n, pad_size + query_num + 4*max_gt_num, 8)  输入decoder前的ref window
            'inter_references_out': inter_references_out,  # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果
            'memory': memory, # 包括此项的以下三项都是用来监督encoder时才会用到的
            'src_anchors': src_anchors,
            'topk_indexes': topk_indexes, # (B_n, query_num, 1) 索引
        }

        fined_query = hs[-1, :, pad_size:pad_size+self.num_queries,:] # (B_n, query_num, 256) 最后一层Decoder layer的输出query
        H, W = src_shape[0,0], src_shape[0,1]

        bboxes_per_layer = bboxes_per_layer[-1, :, pad_size:pad_size+self.num_queries, :] # (B_n, query_num, 8)

        memory_discrete = torch.zeros_like(memory) # (B_n, H*W, 256) 

        memory_discrete = memory_discrete.scatter(1, topk_indexes.repeat(1, 1, memory_discrete.size(-1)), fined_query) # (B_n, H*W, 256) 将query放入到一个空的memory中
        memory_discrete = memory_discrete.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W) # (B_n, C, H, W) 形成稀疏的特征图

        # 新建一个默认参考框，然后将decoder最后一次预测的内容填充进去，这个将会在空间变换后作为分组依据
        boxes_before_trans = copy.deepcopy(src_anchors) # (B_n, HW, 7)
        probs_before_trans = torch.zeros(boxes_before_trans.size(0), boxes_before_trans.size(1), 1).to(boxes_before_trans)
        boxes_before_trans = torch.cat([boxes_before_trans, probs_before_trans], dim=-1) # (B_n, HW, 8)
        boxes_before_trans = boxes_before_trans.scatter(1, topk_indexes.repeat(1, 1, boxes_before_trans.size(-1)), bboxes_per_layer) # (B_n, H*W, 8) 将bbox放入到一个空的特征图中
        boxes_before_trans = boxes_before_trans.permute(0, 2, 1).reshape(memory.shape[0], 8, H, W) # (B_n, 8, H, W) 形成稀疏的特征图

        # 创造mask标记fined query
        valid_flag = torch.ones(fined_query.shape[0], fined_query.shape[1], 1).to(fined_query) # (B_n, query_num, 1) 全1
        memory_mask = torch.zeros(memory.shape[0], memory.shape[1], 1).to(memory) # (B_n, HW, 1)
        memory_mask = memory_mask.scatter(1, topk_indexes.repeat(1, 1, memory_mask.size(-1)), valid_flag) # (B_n, HW, 1)  将fined query给标记
        memory_mask = memory_mask.permute(0, 2, 1).reshape(memory_mask.shape[0], 1, H, W) # (B_n, 1, H, W)

        # 到这里，准备了 1️⃣离散特征图 2️⃣ 离散特征图对应的mask，用来索引和标记 3️⃣ 筛选出来的对应bbox
        memory_discrete_batch_lst = self.regroup(memory_discrete, record_len)
        memory_mask_batch_lst = self.regroup(memory_mask, record_len)
        boxes_before_trans_batch_lst = self.regroup(boxes_before_trans, record_len)

        # memory_batch_lst = self.regroup(memory, record_len)
        all_queries = []
        ref_bboxes = []
        solo_bboxes = []

        com_num_batch = []
        # ===================================性能优化⬇️=====================================================
        # —— START 批量化多 agent、多 query 提取 —— #
        # 已有变量：
        #   record_len:      List[int], B = len(record_len)
        #   memory_discrete_batch_lst: List[Tensor[N_i,C,H,W]]
        #   memory_mask_batch_lst:     List[Tensor[N_i,1,H,W]]
        #   boxes_before_trans_batch_lst: List[Tensor[N_i,8,H,W]]
        #   pairwise_t_matrix:    List[Tensor[N_i,N_i,2,3]]
        #   pairwise_t_matrix_ref:List[Tensor[N_i,N_i,4,4]]
        #   self.box_decode_func, self.box_encode_func, self.pos_embed_layer, box_utils.*

        device = memory_discrete_batch_lst[0].device
        B = len(record_len)
        C, H, W = memory_discrete_batch_lst[0].shape[1:]
        N_max = max(record_len)
        NM = H * W

        # 1) Pad + stack 到大张量
        mem_disc = torch.zeros(B, N_max, C, H, W, device=device)
        mem_mask = torch.zeros(B, N_max, 1, H, W, device=device)
        boxes_bt = torch.zeros(B, N_max, 8, H, W, device=device)
        t_mat    = torch.zeros(B, N_max, N_max, 2, 3, device=device)
        t_ref    = torch.zeros(B, N_max, N_max, 4, 4, device=device)
        for b in range(B):
            Ni = record_len[b]
            mem_disc[b, :Ni] = memory_discrete_batch_lst[b]
            mem_mask[b, :Ni] = memory_mask_batch_lst[b]
            boxes_bt[b, :Ni] = boxes_before_trans_batch_lst[b]
            t_mat[b, :Ni, :Ni] = pairwise_t_matrix[b][:Ni, :Ni]
            t_ref[b, :Ni, :Ni] = pairwise_t_matrix_ref[b][:Ni, :Ni]

        # 2) 批量 warp_affine：
        flat_mem = mem_disc.reshape(-1, C, H, W)                          # (B*N_max, C, H, W)
        flat_t   = t_mat[:, 0, :, :, :].reshape(-1, 2, 3)                # (B*N_max, 2, 3)
        warped   = warp_affine_simple(flat_mem, flat_t, (H, W), mode='nearest')
        neighbor_memory = warped.reshape(B, N_max, C, H, W)              # (B, N_max, C, H, W)

        flat_mask = mem_mask.reshape(-1, 1, H, W)
        warped_m  = warp_affine_simple(flat_mask, flat_t, (H, W), mode='nearest')
        neighbor_mask = warped_m.reshape(B, N_max, 1, H, W)              # (B, N_max, 1, H, W)

        flat_boxes = boxes_bt.reshape(-1, 8, H, W)
        warped_b   = warp_affine_simple(flat_boxes, flat_t, (H, W), mode='nearest')
        neighbor_boxes = warped_b.reshape(B, N_max, 8, H, W)            # (B, N_max, 8, H, W)

        # 3) flatten 空间维度、把通道拉到最后：
        nb_mem = neighbor_memory.flatten(3).permute(0, 1, 3, 2)   # (B, N_max, NM, C)
        nb_msk = neighbor_mask.flatten(3).permute(0, 1, 3, 2).bool()  # (B, N_max, NM, 1)
        nb_box = neighbor_boxes.flatten(3).permute(0, 1, 3, 2)    # (B, N_max, NM, 8)

        # 4) 一次性找出所有“有效 query” 的索引：
        #    idxs 形状 (K, 4)：每行 = [batch_idx, agent_idx, pos_idx, 0]
        idxs = nb_msk.squeeze(-1).nonzero(as_tuple=False)
        batch_idx, agent_idx, pos_idx = idxs[:,0], idxs[:,1], idxs[:,2]

        # 5) 批量 gather 特征 & box：
        feat_k = nb_mem[batch_idx, agent_idx, pos_idx]  # (K, C) 整个batch中的所有agent的feature
        box_k  = nb_box[batch_idx, agent_idx, pos_idx]  # (K, 8)

        # 6) 对 K 条 query 做解码 → 投影 → 重新编码
        #    （这里 K 通常 << N_max*NM，所以这个小循环开销可忽略）
        centers = self.box_decode_func(box_k[:, :7])                     # (K, 7)
        corners = box_utils.boxes_to_corners_3d(centers, 'lwh')          # (K, 8, 3)
        # 每条 query 的变换矩阵 = t_ref[b,a,0] (从 agent a → ego)
        trans_mats = t_ref[batch_idx, agent_idx, 0]                      # (K, 4, 4)

        proj_centers = []
        for i in range(corners.shape[0]): # 遍历K次
            # project_box3d 接受 (1,8,3) + (4,4)
            p3 = box_utils.project_box3d(corners[i:i+1], trans_mats[i]) # 每个参考框依次投影
            pc = box_utils.corners_to_boxes_3d(p3, 'lwh')  # (1,7)
            proj_centers.append(pc)
        proj_centers = torch.cat(proj_centers, dim=0)                   # (K, 7)

        # 7) 重新归一化并拼回 confidence → (K,8)
        norm_centers = self.box_encode_func(proj_centers)               # (K, 7)
        box_norm_k   = torch.cat([norm_centers, box_k[:, 7:]], dim=-1)  # (K, 8)

        # 8) 一次性做位置 embedding
        pos_emb_k = self.pos_embed_layer(box_norm_k) \
                          + self.agent_embed[agent_idx]                 # (K, D)

        # —— 到这里，得到了：
        #     feat_k      (K, C)
        #     pos_emb_k   (K, D)
        #     box_norm_k  (K, 8)

        all_queries = []
        ref_bboxes  = []
        solo_bboxes = []
        com_num_batch = []

        for b in range(B):
            # 1) 挑出本场景的记录
            mask_b    = (batch_idx == b)            # (K,)
            feat_b    = feat_k[mask_b]              # (Mb, C)
            pos_emb_b = pos_emb_k[mask_b]           # (Mb, D)
            box_b     = box_norm_k[mask_b]          # (Mb, 8)
            Mb        = feat_b.shape[0]
            agent_b   = agent_idx[mask_b]           # (Mb,)  选出来对应组内的所有agent的idx

            if Mb == 0:
                # 没 query 的情况
                all_queries.append(None)
                ref_bboxes .append(None)
                solo_bboxes.append(torch.zeros(1,1,8, device=feat_k.device))
                continue

            conf_thresh = 0.1
            # 2) 调用 vectorized Gaussian Mask 函数
            attn_mask_b, fuse_idx_b, indep_idx_list = gaussian_attn_mask_from_bboxes_tensor(
                all_box_norms   = box_b[:, :7],     # (Mb,7)
                all_positions   = box_b[:, :2],     # (Mb,2)
                all_sizes       = box_b[:, 3:5],    # (Mb,2)
                all_confidences = box_b[:, 7],      # (Mb,)
                conf_thresh     = conf_thresh,
                decode_box_func = self.box_decode_func
            )
            # 2.1 根据置信度过滤计算需要发送的query数量
            conf_mask = box_b[:, 7] > conf_thresh       # (Mb,)
            non_ego_mask  = (agent_b > 0)               # (Mb,)
            valid_comm_q  = conf_mask & non_ego_mask    # (Mb,)
            num_q = valid_comm_q.sum().item() # 这一批中总query数
            other = max(1, record_len[b]-1) # 计算有几个non ego agent
            com_num_batch.append(num_q / other)

            # attn_mask_b: (Mf, Mf) 或 None
            # fuse_idx_b:  Tensor([i0, i1, …], dtype=torch.float32)  —— 要融合的那些 indices
            # indep_idx_list: Python list —— 那些“独立”的 idx

            # 3) 做融合（如果有需要融合的 query）
            if attn_mask_b is not None:
                # pick out要融合的特征 & pos
                feat_fuse = feat_b[fuse_idx_b]              # (Mf, C)
                pos_fuse  = pos_emb_b[fuse_idx_b]           # (Mf, D)
                # 调整维度成 (1, Mf, C)/(1, Mf, D)
                fused = self.fd_atten(
                    feat_fuse.unsqueeze(0),
                    pos_fuse .unsqueeze(0),
                    attn_mask_b
                )                                            # (1, Mf, C_out)
                queries_b = fused.squeeze(0)                # (Mf, C_out)
                ref_bboxes_b = box_b[fuse_idx_b, :7]        # (Mf, 7)
            else:
                queries_b   = None
                ref_bboxes_b = None

            # 4) 准备那些“独立检测”不需要交互的 boxes
            if len(indep_idx_list) > 0:
                indep_boxes = box_b[indep_idx_list]        # (Ki, 8)
                # 为了和原接口保持一致，扩一个 batch 维
                indep_boxes = indep_boxes.unsqueeze(0)     # (1, Ki, 8)
            else:
                indep_boxes = torch.zeros(1,1,8, device=feat_k.device)

            # 5) 收集结果
            all_queries.append(queries_b)
            ref_bboxes .append(ref_bboxes_b)
            solo_bboxes.append(indep_boxes)


        """ for bid in range(len(record_len)):
            N = record_len[bid] # number of valid agent
            t_matrix = pairwise_t_matrix[bid][:N, :N, :, :] # (N, N, 2, 3)
            t_matrix_ref = pairwise_t_matrix_ref[bid][:N, :N, :, :] # (N, N, 4, 4)
            select_bbox_b = boxes_before_trans_batch_lst[bid] # (N, 8, H，W) 
            memory_discrete_b = memory_discrete_batch_lst[bid] # (N, C, H, W)
            memory_mask_b = memory_mask_batch_lst[bid] # (N, 1, H, W)

            neighbor_memory = warp_affine_simple(memory_discrete_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, C, H, W)
            neighbor_memory_mask = warp_affine_simple(memory_mask_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 1, H, W)
            neighbor_select_bbox_b = warp_affine_simple(select_bbox_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 8, H，W) 
            
            neighbor_memory = neighbor_memory.flatten(2).permute(0, 2, 1) # (N, HW, C)
            neighbor_memory_mask = neighbor_memory_mask.flatten(2).permute(0, 2, 1) # (N, HW, 1) 这个里面有0有1, 1的地方就是对应其有效的query
            neighbor_select_bbox_b = neighbor_select_bbox_b.flatten(2).permute(0, 2, 1) # (N, HW, 8) 

            neighbor_mask = neighbor_memory_mask.squeeze(-1).bool() # (N, HW)
            valid_query_lst = [neighbor_memory[i][neighbor_mask[i]] for i in range(N)] # [(n1, C), (n2, C)...]
            valid_bbox_lst = [neighbor_select_bbox_b[i][neighbor_mask[i]] for i in range(N)] # [(n1, 8), (n2, 8)...]
            # valid_query_lst = valid_query_lst[1:]
            # valid_bbox_lst = valid_bbox_lst[1:]
            valid_bbox_norm_lst = [] # [(n1, 8), (n2, 8)...] 

            for id in range(len(valid_bbox_lst)):
                valid_box = valid_bbox_lst[id] # (n, 8)
                valid_box_center = self.box_decode_func(valid_box[..., :7]) # (n, 7) 反归一化 变到点云坐标系中的坐标
                valid_box_corner = box_utils.boxes_to_corners_3d(valid_box_center, 'lwh') # (n, 8, 3)
                projected_bbox_corner = box_utils.project_box3d(valid_box_corner.float(), t_matrix_ref[id, 0].float()) # TODO WHY???
                projected_bbox_center = box_utils.corners_to_boxes_3d(projected_bbox_corner, 'lwh') # (n, 7)
                projected_bbox_center_norm = self.box_encode_func(projected_bbox_center) # 重新归一化
                # print("t_matrix_ref[0, id] is ", t_matrix_ref[0, id])
                # print("t_matrix_ref[id, 0] is ", t_matrix_ref[id, 0])
                # xxx
                # projected_bbox_center = torch.cat([projected_bbox_center, valid_box[:, 7:]], dim=-1) # # (n, 8)
                projected_bbox_center_norm = torch.cat([projected_bbox_center_norm, valid_box[:, 7:]], dim=-1) # # (n, 8)
                # projected_bbox_center_norm = valid_box 

                # valid_bbox_lst[id] = projected_bbox_center # 到这里后所有的box都统一到ego坐标系了 且所有的box都是真实坐标系，非归一化数值
                valid_bbox_norm_lst.append(projected_bbox_center_norm)

            # neighbor_index = torch.nonzero(neighbor_mask, as_tuple=False) # (N, HW)
                
            # 生成网格索引
            i_indices = torch.arange(H, device=neighbor_mask.device).repeat(W).view(1, -1)  # (1, HW) 每H个元素复制一遍，复制W遍
            j_indices = torch.arange(W, device=neighbor_mask.device).repeat_interleave(H).view(1, -1)  # (1, HW) # 这是每个元素复制H遍
            # 扩展索引以匹配批次大小
            i_indices = i_indices.expand(N, -1)  # (N, HW)
            j_indices = j_indices.expand(N, -1)  # (N, HW)

            # 提取有效位置的索引
            # valid_i = i_indices[neighbor_mask == 1]  
            # valid_j = j_indices[neighbor_mask == 1]  # 所有有效位置的 j 坐标

            query_info_lst = []
            com_num = 0
            for i in range(len(valid_query_lst)): # 遍历每个agent
                n_q = valid_query_lst[i].size(0)
                agent_queries = valid_query_lst[i] # (n, 8)
                # agent_bboxes = valid_bbox_lst[i] # (n, 8)
                agent_bboxes_norm = valid_bbox_norm_lst[i] # (n,8)
                agent_pos_emb = self.pos_embed_layer(agent_bboxes_norm)
                
                valid_mask  = neighbor_mask[i] # (HW,)
                valid_i = i_indices[i][valid_mask == 1] # 所有有效位置的 i 坐标 (n, )
                valid_j = j_indices[i][valid_mask == 1] # 所有有效位置的 j 坐标
                valid_2d_pos = torch.stack([valid_i, valid_j], dim=-1) # (n, 2)
                for j in range(n_q): # 遍历每个query
                    query_info = {
                        "agent_id": i,
                        "box_norm": agent_bboxes_norm[j][:7], # （7）
                        "position": agent_bboxes_norm[j][:2], # (2) cx, cy
                        "bbox_size": agent_bboxes_norm[j][3:5], # (2) l, w
                        # "heading": agent_bboxes[j][6:7],
                        "2d_pos": valid_2d_pos[j], # (2,) 2d坐标
                        "confidence": agent_bboxes_norm[j][7:],
                        "pos_emb": agent_pos_emb[j], # 256
                        "feature": agent_queries[j]
                    }
                    # if i > 0: # 消融实验
                    #     if agent_bboxes_norm[j][7:] > 0.1:
                    #         query_info_lst.append(query_info)
                    # else:
                    query_info_lst.append(query_info)

                    # 计算通信量 也就是传输的query 个数
                    if i > 0:
                        if agent_bboxes_norm[j][7:] > 0.1:
                            com_num += 1
            extra_agents_num = max(1, N-1)
            com_num /= extra_agents_num
            com_num_batch.append(com_num)

            # 🌟 我们的主张是 将所有的query对应的box放在一起，判断两两iou，如果有某个和其他所有box的重合度都为0.1或者以下，认为它是独立检测，则这个不需要交互，直接参与最后的匹配
            attn_mask, valid_indicies, indep_queries = gaussian_atten_mask_from_bboxes(query_info_lst, decode_box_func=self.box_decode_func) # (M, M)的Mask
            # attn_mask = None
            if attn_mask is not None:
                valid_feat = []
                valid_feat_pos = []
                norm_bboxes = []
                for vid in valid_indicies:
                    per_query_feat = query_info_lst[vid]['feature']
                    # per_query_feat_w_pos = query_info_lst[vid]['feature'] + query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_pos = query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_box = query_info_lst[vid]['box_norm']

                    valid_feat.append(per_query_feat.unsqueeze(0)) # (1, D)
                    valid_feat_pos.append(per_query_pos.unsqueeze(0)) # (1, D)
                    norm_bboxes.append(per_query_box.unsqueeze(0)) # (1, 7)
                valid_feat = torch.cat(valid_feat, dim=0).unsqueeze(0) # (1, M, D)
                valid_feat_pos = torch.cat(valid_feat_pos, dim=0).unsqueeze(0) # (1, M, D)
                norm_bboxes = torch.cat(norm_bboxes, dim=0) # (M, 7)

                fused_query = self.fd_atten(valid_feat, valid_feat_pos, attn_mask)

                queries = fused_query.squeeze(0) # n_all, 256

                ref_bbox = norm_bboxes # n_all, 7
            else:
                queries = ref_bbox = None

            if len(indep_queries) > 0:
                indep_boxes = []
                for indep_query in indep_queries:
                    indep_box = torch.cat((indep_query['box_norm'], indep_query['confidence']), dim=-1) # (8)
                    indep_boxes.append(indep_box.unsqueeze(0))
                indep_boxes = torch.cat(indep_boxes, dim=0) # (indep_num, 8)
                indep_boxes = indep_boxes.unsqueeze(0) # (1, indep_num, 8)
            else:
                indep_boxes = None
                indep_boxes = torch.zeros(1, 1, 8).to(memory)

            all_queries.append(queries)
            ref_bboxes.append(ref_bbox)
            solo_bboxes.append(indep_boxes)
            
            self.sample_idx += 1 """

        return result, all_queries, ref_bboxes, solo_bboxes, com_num_batch

class TransIFFV1(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        nlevel=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=1,
        mom=0.999,
        cp_flag=False,
        box_encode_func=None, 
        box_decode_func=None, 
        get_sparse_features_func=None,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom

        self.box_encode_func=box_encode_func
        self.box_decode_func=box_decode_func
        self.get_sparse_features_func=get_sparse_features_func

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)
        self.fd_atten = Fusion_Decoder_CDA(d_model)
        self.merge_linear = nn.Linear(d_model, d_model)

        self.agent_embed = nn.Parameter(torch.Tensor(2, d_model))
        self.pos_embed_layer = MLP(8, d_model, d_model, 3)
        self.rel_pos_mlp   = MLP(2, d_model, d_model, 2)    # 输入 2 维 (i,j)，输出 d_model
        self.merge_linear  = nn.Linear(d_model*2, d_model)   # 用于合并同一位置的 features
        self.sample_idx = 0
        self.parameters_fix()

    def parameters_fix(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False


    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025  # 0.01 - 0.05 w.r.t. Deform-DETR
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1).expand(
                B, -1, -1
            )

            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _get_enc_proposals(self, enc_embed, ref_windows, indexes=None, heatmap=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows = self.proposal_head(enc_embed, ref_windows)

        out_probs = out_logits[..., 0].sigmoid()
        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False)
        topk_probs = topk_probs.unsqueeze(-1)
        indexes = indexes.unsqueeze(-1)
        # print("out_probs  is ", [round(x, 3) for x in out_probs[0][:1000].tolist()])

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs.detach().expand(-1, -1, out_logits.shape[-1]),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None, heatmap=None):
        '''
        ⚡ 先自车检测， 获得高质量query后传输
        src: [(B_n, 256, H, W)]
        pos: [(B_n, 256, H, W)]
        noised_gt_box: (B_n, pad_size, 7)  这里用的应该是single gt 因为这个要先refine单车 形成优质query
        noised_gt_onehot: (B_n, pad_size, num_classes)
        attn_mask: (1000+pad_size, 1000+pad_size)
        targets: [{'gt_boxes': (N, 7), 'labels': (N, )}, ...]
        '''
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src) # 创造参考框，这个是BoxAttention必须的 (B_n, HW, 7)
        src, _, src_shape = flatten_with_shape(src, None)# 展平特征图，返回的是 (B_n, H*W, 256), None, (1, 2) 最后一项记录着H，W 即feature shape
        src_pos = []
        for pe in pos:
            B, C = pe.shape[:2]
            pe = pe.view(B, C, -1).transpose(1, 2) # b, h*w, c
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1) # (B_n, H*W, C)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]) # 这是为了生成划分的索引，区分每个特征图的位置，由于只有一个特征图，所以结果是(0,)

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors) # BoxAttention 提取特征 结果为(B_n, H*W, 256)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors, heatmap=heatmap) # 返回None，None，(B_n, query_num, 8)，(B_n, query_num, 1)
        
        pad_size = 0
        # 加噪声gt，准备一起参与decoder训练去噪
        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            ) # (B_n, pad_size, 8)
            pad_size = noised_gt_proposals.size(1)
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            ) # (B_n, pad_size + query_num, 8) 
        init_reference_out = topk_proposals[..., :7]

        # hs, inter_references = self.decoder_gt(
        hs, inter_references, bboxes_per_layer = self.decoder(
            query_embed, # None 
            query_pos, # None
            memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
            src_shape, # (1, 2)
            src_start_index, # (0,)
            topk_proposals, # (B, query_num, 8)
            attn_mask,
            return_bboxes=True
        ) # (3, B_n, pad_size + query_num, 256) 每一层的输出的query特征， (3， B_n, pad_size + all_query_num, 7) 每一层的检测结果 

        # optional gt forward 对比学习需要用到的动量更新模型用加噪gt来做对比学习的
        if targets is not None:
            batch_size = len(targets) # 这里是single 标签
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets] # [N1, N2, N3, N4] 此为B=4时的各个样本的GT数
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, 8) # (B, max_gt_num, 8)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :7] = targets[bi]["gt_boxes"] # 放入gt的box 和 one-hot 分类编码
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], 7:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder() # 动量更新辅助模型，其参数更新速度非常缓慢，但一直追随decoder
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2) # 得到去噪gt组数 == 3  2指的是每一组又分正负样本
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    ) # 每一组抽取max_gt_num个 (B_n, 3*max_gt_num, 8) 这是相当于去噪正样本抽取出来
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()  # （4*max_gt_num，4*max_gt_num）全True
                    for di in range(dn_group_num + 1): # 对角部分mask 全部设置为False，相当于说只关注自己，即每一批gt，无论有无噪声，仅关注自身，屏蔽组之间的可见性
                        gt_attn_mask[
                            di * max_gt_num : (di + 1) * max_gt_num,
                            di * max_gt_num : (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt( # 辅助模型进行对比学习，缓慢追随decoder。 返回 (3，B_n, 4*max_gt_num, 256) 与 (3，B_n, 4*max_gt_num, 8)
                    None,
                    None,
                    memory, # BoxAttention 提取特征后结合多agent后的Feature Map 结果为(B_n, H*W, 256)
                    src_shape, # (1, 2)
                    src_start_index, # (0,)
                    gt_proposals, # (B_n, 4*max_gt_num, 8)
                    gt_attn_mask, #（4*max_gt_num，4*max_gt_num）
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :7],
                ),
                dim=1,
            ) # (B_n, pad_size + query_num + 4*max_gt_num, 7)  输入decoder前的ref window

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            ) # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            ) # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果

        inter_references_out = inter_references
        '''
        从前往后依次返回: Decoder layer每一层的query, 输入Decoder的参考框, Decoder layer每一层的检测结果, Encoder输出的特征图, 初始化的参考框, ego的最高query_num的索引
        TODO Encoder输出的特征图信息会不会不足? 要不要考虑将query融合后的信息放回去 🌟Updated: Done, 先看看性能
        '''
        result = {
            'hs':hs, # (3, B_n, pad_size + query_num + 4*max_gt_num, 256) 每一层Decoder layer的输出query
            'init_reference_out': init_reference_out,  # (B_n, pad_size + query_num + 4*max_gt_num, 8)  输入decoder前的ref window
            'inter_references_out': inter_references_out,  # (3，B_n, pad_size + query_num + 4*max_gt_num, 7) 每一层Decoder layer的对应检测结果
            'memory': memory, # 包括此项的以下三项都是用来监督encoder时才会用到的
            'src_anchors': src_anchors,
            'topk_indexes': topk_indexes, # (B_n, query_num, 1) 索引
        }

        fined_query = hs[-1, :, pad_size:pad_size+self.num_queries,:] # (B_n, query_num, 256) 最后一层Decoder layer的输出query
        H, W = src_shape[0,0], src_shape[0,1]

        bboxes_per_layer = bboxes_per_layer[-1, :, pad_size:pad_size+self.num_queries, :] # (B_n, query_num, 8)

        memory_discrete = torch.zeros_like(memory) # (B_n, H*W, 256) 

        memory_discrete = memory_discrete.scatter(1, topk_indexes.repeat(1, 1, memory_discrete.size(-1)), fined_query) # (B_n, H*W, 256) 将query放入到一个空的memory中
        memory_discrete = memory_discrete.permute(0, 2, 1).reshape(memory.shape[0], memory.shape[-1], H, W) # (B_n, C, H, W) 形成稀疏的特征图

        # 新建一个默认参考框，然后将decoder最后一次预测的内容填充进去，这个将会在空间变换后作为分组依据
        boxes_before_trans = copy.deepcopy(src_anchors) # (B_n, HW, 7)
        probs_before_trans = torch.zeros(boxes_before_trans.size(0), boxes_before_trans.size(1), 1).to(boxes_before_trans)
        boxes_before_trans = torch.cat([boxes_before_trans, probs_before_trans], dim=-1) # (B_n, HW, 8)
        boxes_before_trans = boxes_before_trans.scatter(1, topk_indexes.repeat(1, 1, boxes_before_trans.size(-1)), bboxes_per_layer) # (B_n, H*W, 8) 将bbox放入到一个空的特征图中
        boxes_before_trans = boxes_before_trans.permute(0, 2, 1).reshape(memory.shape[0], 8, H, W) # (B_n, 8, H, W) 形成稀疏的特征图

        # 创造mask标记fined query
        valid_flag = torch.ones(fined_query.shape[0], fined_query.shape[1], 1).to(fined_query) # (B_n, query_num, 1) 全1
        memory_mask = torch.zeros(memory.shape[0], memory.shape[1], 1).to(memory) # (B_n, HW, 1)
        memory_mask = memory_mask.scatter(1, topk_indexes.repeat(1, 1, memory_mask.size(-1)), valid_flag) # (B_n, HW, 1)  将fined query给标记
        memory_mask = memory_mask.permute(0, 2, 1).reshape(memory_mask.shape[0], 1, H, W) # (B_n, 1, H, W)

        # 到这里，准备了 1️⃣离散特征图 2️⃣ 离散特征图对应的mask，用来索引和标记 3️⃣ 筛选出来的对应bbox
        memory_discrete_batch_lst = self.regroup(memory_discrete, record_len)
        memory_mask_batch_lst = self.regroup(memory_mask, record_len)
        boxes_before_trans_batch_lst = self.regroup(boxes_before_trans, record_len)

        # memory_batch_lst = self.regroup(memory, record_len)
        all_queries = []
        ref_bboxes = []
        solo_bboxes = []

        com_num_batch = []
        # ===================================性能优化⬇️=====================================================
        # —— START 批量化多 agent、多 query 提取 —— #
        # 已有变量：
        #   record_len:      List[int], B = len(record_len)
        #   memory_discrete_batch_lst: List[Tensor[N_i,C,H,W]]
        #   memory_mask_batch_lst:     List[Tensor[N_i,1,H,W]]
        #   boxes_before_trans_batch_lst: List[Tensor[N_i,8,H,W]]
        #   pairwise_t_matrix:    List[Tensor[N_i,N_i,2,3]]
        #   pairwise_t_matrix_ref:List[Tensor[N_i,N_i,4,4]]
        #   self.box_decode_func, self.box_encode_func, self.pos_embed_layer, box_utils.*

        device = memory_discrete_batch_lst[0].device
        B = len(record_len)
        C, H, W = memory_discrete_batch_lst[0].shape[1:]
        N_max = max(record_len)
        NM = H * W

        # 1) Pad + stack 到大张量
        mem_disc = torch.zeros(B, N_max, C, H, W, device=device)
        mem_mask = torch.zeros(B, N_max, 1, H, W, device=device)
        boxes_bt = torch.zeros(B, N_max, 8, H, W, device=device)
        t_mat    = torch.zeros(B, N_max, N_max, 2, 3, device=device)
        t_ref    = torch.zeros(B, N_max, N_max, 4, 4, device=device)
        for b in range(B):
            Ni = record_len[b]
            mem_disc[b, :Ni] = memory_discrete_batch_lst[b]
            mem_mask[b, :Ni] = memory_mask_batch_lst[b]
            boxes_bt[b, :Ni] = boxes_before_trans_batch_lst[b]
            t_mat[b, :Ni, :Ni] = pairwise_t_matrix[b][:Ni, :Ni]
            t_ref[b, :Ni, :Ni] = pairwise_t_matrix_ref[b][:Ni, :Ni]

        # 2) 批量 warp_affine：
        flat_mem = mem_disc.reshape(-1, C, H, W)                          # (B*N_max, C, H, W)
        flat_t   = t_mat[:, 0, :, :, :].reshape(-1, 2, 3)                # (B*N_max, 2, 3)
        warped   = warp_affine_simple(flat_mem, flat_t, (H, W), mode='nearest')
        neighbor_memory = warped.reshape(B, N_max, C, H, W)              # (B, N_max, C, H, W)

        flat_mask = mem_mask.reshape(-1, 1, H, W)
        warped_m  = warp_affine_simple(flat_mask, flat_t, (H, W), mode='nearest')
        neighbor_mask = warped_m.reshape(B, N_max, 1, H, W)              # (B, N_max, 1, H, W)

        flat_boxes = boxes_bt.reshape(-1, 8, H, W)
        warped_b   = warp_affine_simple(flat_boxes, flat_t, (H, W), mode='nearest')
        neighbor_boxes = warped_b.reshape(B, N_max, 8, H, W)            # (B, N_max, 8, H, W)

        # 3) flatten 空间维度、把通道拉到最后：
        nb_mem = neighbor_memory.flatten(3).permute(0, 1, 3, 2)   # (B, N_max, NM, C)
        nb_msk = neighbor_mask.flatten(3).permute(0, 1, 3, 2).bool()  # (B, N_max, NM, 1)
        nb_box = neighbor_boxes.flatten(3).permute(0, 1, 3, 2)    # (B, N_max, NM, 8)

        # 4) 一次性找出所有“有效 query” 的索引：
        #    idxs 形状 (K, 4)：每行 = [batch_idx, agent_idx, pos_idx, 0]
        idxs = nb_msk.squeeze(-1).nonzero(as_tuple=False)
        batch_idx, agent_idx, pos_idx = idxs[:,0], idxs[:,1], idxs[:,2]

        # 5) 批量 gather 特征 & box：
        feat_k = nb_mem[batch_idx, agent_idx, pos_idx]  # (K, C) 整个batch中的所有agent的feature
        box_k  = nb_box[batch_idx, agent_idx, pos_idx]  # (K, 8)

        # 6) 对 K 条 query 做解码 → 投影 → 重新编码
        #    （这里 K 通常 << N_max*NM，所以这个小循环开销可忽略）
        centers = self.box_decode_func(box_k[:, :7])                     # (K, 7)
        corners = box_utils.boxes_to_corners_3d(centers, 'lwh')          # (K, 8, 3)
        # 每条 query 的变换矩阵 = t_ref[b,a,0] (从 agent a → ego)
        trans_mats = t_ref[batch_idx, agent_idx, 0]                      # (K, 4, 4)

        proj_centers = []
        for i in range(corners.shape[0]): # 遍历K次
            # project_box3d 接受 (1,8,3) + (4,4)
            p3 = box_utils.project_box3d(corners[i:i+1], trans_mats[i]) # 每个参考框依次投影
            pc = box_utils.corners_to_boxes_3d(p3, 'lwh')  # (1,7)
            proj_centers.append(pc)
        proj_centers = torch.cat(proj_centers, dim=0)                   # (K, 7)

        # 7) 重新归一化并拼回 confidence → (K,8)
        norm_centers = self.box_encode_func(proj_centers)               # (K, 7)
        box_norm_k   = torch.cat([norm_centers, box_k[:, 7:]], dim=-1)  # (K, 8)

        # 8) 一次性做位置 embedding
        # pos_emb_k = self.pos_embed_layer(box_norm_k) \
        #                   + self.agent_embed[agent_idx]                 # (K, D)
        pos_emb_k = self.pos_embed_layer(box_norm_k)                      # (K, D)

        # —— 到这里，得到了：
        #     feat_k      (K, C)
        #     pos_emb_k   (K, D)
        #     box_norm_k  (K, 8)

        all_queries = []
        ref_bboxes  = []
        solo_bboxes = []
        com_num_batch = []

        for b in range(B):
            # 1) 挑出本场景的记录
            mask_b    = (batch_idx == b)            # (K,)
            feat_b    = feat_k[mask_b]              # (Mb, C)
            pos_emb_b = pos_emb_k[mask_b]           # (Mb, D)
            box_b     = box_norm_k[mask_b]          # (Mb, 8)
            pos_idx_b  = pos_idx[mask_b]              # (Mb,) 一维索引 = i*W + j
            Mb        = feat_b.shape[0]
            agent_b   = agent_idx[mask_b]           # (Mb,)  选出来对应组内的所有agent的idx

            if Mb == 0:
                # 没 query 的情况
                all_queries.append(None)
                ref_bboxes .append(None)
                solo_bboxes.append(torch.zeros(1,1,8, device=feat_k.device))
                continue

            # ---- 2) 2D 相对位置编码 ---- #
            # 还原出行列 i,j，并归一化到 [0,1]
            i = (pos_idx_b // W).unsqueeze(1).float() / (H-1)
            j = (pos_idx_b %  W).unsqueeze(1).float() / (W-1)
            rel_pos = torch.cat([i, j], dim=1)           # (Mv, 2)
            rel_emb = self.rel_pos_mlp(rel_pos)          # (Mv, D)

            # filter_mask = box_b[:, 7] > conf_thresh      # (Mb,) 置信度筛选
            # if not filter_mask.any():
            #     all_queries.append(None)
            #     ref_bboxes .append(None)
            #     solo_bboxes.append(torch.zeros(1, 0, 8, device=feat_k.device))
            #     continue

            # feat_b = feat_b[filter_mask]                 # (Mv, C)
            # pos_emb_b0  = pos_emb_b[filter_mask]         # (Mv, D)
            # box_b       = box_b      [filter_mask]        # (Mv, 8)
            # pos_idx_b   = pos_idx_b  [filter_mask]        # (Mv,)
            # agent_b     = agent_b    [filter_mask]        # (Mv,)

            conf_thresh = 0.1
            # 2) 调用 vectorized Gaussian Mask 函数
            attn_mask_b, fuse_idx_b, indep_idx_list = gaussian_attn_mask_from_bboxes_tensor(
                all_box_norms   = box_b[:, :7],     # (Mb,7)
                all_positions   = box_b[:, :2],     # (Mb,2)
                all_sizes       = box_b[:, 3:5],    # (Mb,2)
                all_confidences = box_b[:, 7],      # (Mb,)
                conf_thresh     = conf_thresh,
                decode_box_func = self.box_decode_func,
                iou_thresh     = 0.0    # 这里设置为1则不使用iou进行筛选即所有的query都是需要fused，先CDA再FM
            )
            # 2.1 根据置信度过滤计算需要发送的query数量
            conf_mask = box_b[:, 7] > conf_thresh       # (Mb,)
            non_ego_mask  = (agent_b > 0)               # (Mb,)
            valid_comm_q  = conf_mask & non_ego_mask    # (Mb,)
            num_q = valid_comm_q.sum().item() # 这一批中总query数
            other = max(1, record_len[b]-1) # 计算有几个non ego agent
            com_num_batch.append(num_q / other)

            # attn_mask_b: (Mf, Mf) 或 None
            # fuse_idx_b:  Tensor([i0, i1, …], dtype=torch.float32)  —— 要融合的那些 indices
            # indep_idx_list: Python list —— 那些“独立”的 idx

            # print("fuse_idx_b is", fuse_idx_b)

            feat_fuse = feat_b[fuse_idx_b]              # (Mf, C) #  不缺分soloboxes 因为我们设置了iou_thresh=1.1
            pos_fuse  = pos_emb_b[fuse_idx_b]           # (Mf, C)
            # print("feat_fuse shape is", feat_fuse.shape)
            # 调整维度成 (1, Mf, C)/(1, Mf, D)
            fused = self.fd_atten(
                feat_fuse.unsqueeze(0),
                pos_fuse .unsqueeze(0),
                attn_mask_b
            )                                            # (1, Mf, C_out)
            fused = fused.squeeze(0)
            # ---- 4) 按 2D 位置合并同一格子的 queries ---- #
            # key = pos_idx_b，只同一 pos_idx 聚为一组
            box_filter_b = box_b[fuse_idx_b]                      # (Mf, 8)
            pos_idx_filter_b = pos_idx_b[fuse_idx_b]              # (Mf, )  
            uniq_pos, inv = torch.unique(pos_idx_filter_b, return_inverse=True) # (Mf_uniq,), (Mf) 分别是去重后的元素以及原来每个元素在这个去重结果里的索引
            G = uniq_pos.size(0)
            merged_feats = []
            merged_boxes = []
            for gi in range(G): # 遍历去重后的每一个的索引
                mask_g = (inv == gi) # (Mf, ) 看看重复的地方
                grp_feats = feat_fuse[mask_g]        # (n_g, C) 应该是
                grp_boxes = box_filter_b[mask_g]     # (n_g, 8)
                # m_feat = self.merge_linear(grp_feats)
                n_g = grp_feats.size(0)
                if n_g == 1:
                    # 只有 1 条，直接用原特征／box
                    m_feat = grp_feats                     # (1, D)
                    m_box  = grp_boxes                    # (1, 8)

                elif n_g == 2:
                    # 刚好 2 条，先 concat 再线性映射回来
                    # grp_feats[0]、grp_feats[1] 各 (D,)
                    cat = torch.cat([grp_feats[0], grp_feats[1]], dim=-1).unsqueeze(0)  # (1, 2D)
                    m_feat = self.merge_linear(cat)                                   # (1, D)
                    # boxes 这一维用平均
                    m_box = grp_boxes.mean(0, keepdim=True)                           # (1, 8)

                else:
                    # 超过 2 条，直接 mean
                    m_feat = grp_feats.mean(0, keepdim=True)                         # (1, D)
                    m_box  = grp_boxes.mean(0, keepdim=True)                          # (1, 8)
                merged_feats.append(m_feat)
                merged_boxes.append(m_box)
            if len(merged_feats) > 0:
                queries_b   = torch.cat(merged_feats, dim=0)   # (G, D)
                ref_bboxes_b = torch.cat(merged_boxes, dim=0)[:, :7]  # (G, 7)
            else:
                queries_b   = None
                ref_bboxes_b = None
            # # 3) 做CDA融合（如果有需要融合的 query）
            # if attn_mask_b is not None:
            #     # pick out要融合的特征 & pos
            #     feat_fuse = feat_b[fuse_idx_b]              # (Mf, C)
            #     pos_fuse  = pos_emb_b[fuse_idx_b]           # (Mf, D)
            #     # 调整维度成 (1, Mf, C)/(1, Mf, D)
            #     fused = self.fd_atten(
            #         feat_fuse.unsqueeze(0),
            #         pos_fuse .unsqueeze(0),
            #         attn_mask_b
            #     )                                            # (1, Mf, C_out)
            #     queries_b = fused.squeeze(0)                # (Mf, C_out)
            #     ref_bboxes_b = box_b[fuse_idx_b, :7]        # (Mf, 7)
            # else:
            #     queries_b   = None
            #     ref_bboxes_b = None

            # 4) 准备那些“独立检测”不需要交互的 boxes
            if len(indep_idx_list) > 0:
                indep_boxes = box_b[indep_idx_list]        # (Ki, 8)
                # 为了和原接口保持一致，扩一个 batch 维
                indep_boxes = indep_boxes.unsqueeze(0)     # (1, Ki, 8)
            else:
                indep_boxes = torch.zeros(1,1,8, device=feat_k.device)

            # 5) 收集结果
            all_queries.append(queries_b)
            ref_bboxes .append(ref_bboxes_b)
            solo_bboxes.append(indep_boxes)


        """ for bid in range(len(record_len)):
            N = record_len[bid] # number of valid agent
            t_matrix = pairwise_t_matrix[bid][:N, :N, :, :] # (N, N, 2, 3)
            t_matrix_ref = pairwise_t_matrix_ref[bid][:N, :N, :, :] # (N, N, 4, 4)
            select_bbox_b = boxes_before_trans_batch_lst[bid] # (N, 8, H，W) 
            memory_discrete_b = memory_discrete_batch_lst[bid] # (N, C, H, W)
            memory_mask_b = memory_mask_batch_lst[bid] # (N, 1, H, W)

            neighbor_memory = warp_affine_simple(memory_discrete_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, C, H, W)
            neighbor_memory_mask = warp_affine_simple(memory_mask_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 1, H, W)
            neighbor_select_bbox_b = warp_affine_simple(select_bbox_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 8, H，W) 
            
            neighbor_memory = neighbor_memory.flatten(2).permute(0, 2, 1) # (N, HW, C)
            neighbor_memory_mask = neighbor_memory_mask.flatten(2).permute(0, 2, 1) # (N, HW, 1) 这个里面有0有1, 1的地方就是对应其有效的query
            neighbor_select_bbox_b = neighbor_select_bbox_b.flatten(2).permute(0, 2, 1) # (N, HW, 8) 

            neighbor_mask = neighbor_memory_mask.squeeze(-1).bool() # (N, HW)
            valid_query_lst = [neighbor_memory[i][neighbor_mask[i]] for i in range(N)] # [(n1, C), (n2, C)...]
            valid_bbox_lst = [neighbor_select_bbox_b[i][neighbor_mask[i]] for i in range(N)] # [(n1, 8), (n2, 8)...]
            # valid_query_lst = valid_query_lst[1:]
            # valid_bbox_lst = valid_bbox_lst[1:]
            valid_bbox_norm_lst = [] # [(n1, 8), (n2, 8)...] 

            for id in range(len(valid_bbox_lst)):
                valid_box = valid_bbox_lst[id] # (n, 8)
                valid_box_center = self.box_decode_func(valid_box[..., :7]) # (n, 7) 反归一化 变到点云坐标系中的坐标
                valid_box_corner = box_utils.boxes_to_corners_3d(valid_box_center, 'lwh') # (n, 8, 3)
                projected_bbox_corner = box_utils.project_box3d(valid_box_corner.float(), t_matrix_ref[id, 0].float()) # TODO WHY???
                projected_bbox_center = box_utils.corners_to_boxes_3d(projected_bbox_corner, 'lwh') # (n, 7)
                projected_bbox_center_norm = self.box_encode_func(projected_bbox_center) # 重新归一化
                # print("t_matrix_ref[0, id] is ", t_matrix_ref[0, id])
                # print("t_matrix_ref[id, 0] is ", t_matrix_ref[id, 0])
                # xxx
                # projected_bbox_center = torch.cat([projected_bbox_center, valid_box[:, 7:]], dim=-1) # # (n, 8)
                projected_bbox_center_norm = torch.cat([projected_bbox_center_norm, valid_box[:, 7:]], dim=-1) # # (n, 8)
                # projected_bbox_center_norm = valid_box 

                # valid_bbox_lst[id] = projected_bbox_center # 到这里后所有的box都统一到ego坐标系了 且所有的box都是真实坐标系，非归一化数值
                valid_bbox_norm_lst.append(projected_bbox_center_norm)

            # neighbor_index = torch.nonzero(neighbor_mask, as_tuple=False) # (N, HW)
                
            # 生成网格索引
            i_indices = torch.arange(H, device=neighbor_mask.device).repeat(W).view(1, -1)  # (1, HW) 每H个元素复制一遍，复制W遍
            j_indices = torch.arange(W, device=neighbor_mask.device).repeat_interleave(H).view(1, -1)  # (1, HW) # 这是每个元素复制H遍
            # 扩展索引以匹配批次大小
            i_indices = i_indices.expand(N, -1)  # (N, HW)
            j_indices = j_indices.expand(N, -1)  # (N, HW)

            # 提取有效位置的索引
            # valid_i = i_indices[neighbor_mask == 1]  
            # valid_j = j_indices[neighbor_mask == 1]  # 所有有效位置的 j 坐标

            query_info_lst = []
            com_num = 0
            for i in range(len(valid_query_lst)): # 遍历每个agent
                n_q = valid_query_lst[i].size(0)
                agent_queries = valid_query_lst[i] # (n, 8)
                # agent_bboxes = valid_bbox_lst[i] # (n, 8)
                agent_bboxes_norm = valid_bbox_norm_lst[i] # (n,8)
                agent_pos_emb = self.pos_embed_layer(agent_bboxes_norm)
                
                valid_mask  = neighbor_mask[i] # (HW,)
                valid_i = i_indices[i][valid_mask == 1] # 所有有效位置的 i 坐标 (n, )
                valid_j = j_indices[i][valid_mask == 1] # 所有有效位置的 j 坐标
                valid_2d_pos = torch.stack([valid_i, valid_j], dim=-1) # (n, 2)
                for j in range(n_q): # 遍历每个query
                    query_info = {
                        "agent_id": i,
                        "box_norm": agent_bboxes_norm[j][:7], # （7）
                        "position": agent_bboxes_norm[j][:2], # (2) cx, cy
                        "bbox_size": agent_bboxes_norm[j][3:5], # (2) l, w
                        # "heading": agent_bboxes[j][6:7],
                        "2d_pos": valid_2d_pos[j], # (2,) 2d坐标
                        "confidence": agent_bboxes_norm[j][7:],
                        "pos_emb": agent_pos_emb[j], # 256
                        "feature": agent_queries[j]
                    }
                    # if i > 0: # 消融实验
                    #     if agent_bboxes_norm[j][7:] > 0.1:
                    #         query_info_lst.append(query_info)
                    # else:
                    query_info_lst.append(query_info)

                    # 计算通信量 也就是传输的query 个数
                    if i > 0:
                        if agent_bboxes_norm[j][7:] > 0.1:
                            com_num += 1
            extra_agents_num = max(1, N-1)
            com_num /= extra_agents_num
            com_num_batch.append(com_num)

            # 🌟 我们的主张是 将所有的query对应的box放在一起，判断两两iou，如果有某个和其他所有box的重合度都为0.1或者以下，认为它是独立检测，则这个不需要交互，直接参与最后的匹配
            attn_mask, valid_indicies, indep_queries = gaussian_atten_mask_from_bboxes(query_info_lst, decode_box_func=self.box_decode_func) # (M, M)的Mask
            # attn_mask = None
            if attn_mask is not None:
                valid_feat = []
                valid_feat_pos = []
                norm_bboxes = []
                for vid in valid_indicies:
                    per_query_feat = query_info_lst[vid]['feature']
                    # per_query_feat_w_pos = query_info_lst[vid]['feature'] + query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_pos = query_info_lst[vid]['pos_emb'] + self.agent_embed[query_info_lst[vid]['agent_id']]
                    per_query_box = query_info_lst[vid]['box_norm']

                    valid_feat.append(per_query_feat.unsqueeze(0)) # (1, D)
                    valid_feat_pos.append(per_query_pos.unsqueeze(0)) # (1, D)
                    norm_bboxes.append(per_query_box.unsqueeze(0)) # (1, 7)
                valid_feat = torch.cat(valid_feat, dim=0).unsqueeze(0) # (1, M, D)
                valid_feat_pos = torch.cat(valid_feat_pos, dim=0).unsqueeze(0) # (1, M, D)
                norm_bboxes = torch.cat(norm_bboxes, dim=0) # (M, 7)

                fused_query = self.fd_atten(valid_feat, valid_feat_pos, attn_mask)

                queries = fused_query.squeeze(0) # n_all, 256

                ref_bbox = norm_bboxes # n_all, 7
            else:
                queries = ref_bbox = None

            if len(indep_queries) > 0:
                indep_boxes = []
                for indep_query in indep_queries:
                    indep_box = torch.cat((indep_query['box_norm'], indep_query['confidence']), dim=-1) # (8)
                    indep_boxes.append(indep_box.unsqueeze(0))
                indep_boxes = torch.cat(indep_boxes, dim=0) # (indep_num, 8)
                indep_boxes = indep_boxes.unsqueeze(0) # (1, indep_num, 8)
            else:
                indep_boxes = None
                indep_boxes = torch.zeros(1, 1, 8).to(memory)

            all_queries.append(queries)
            ref_bboxes.append(ref_bbox)
            solo_bboxes.append(indep_boxes)
            
            self.sample_idx += 1 """

        return result, all_queries, ref_bboxes, solo_bboxes, com_num_batch

def gaussian_atten_mask_from_bboxes(all_queries, conf_thresh=0.1, decode_box_func = None):

    # 1) 先过滤置信度
    valid_indices = [i for i, q in enumerate(all_queries) if q['confidence'] >= conf_thresh]

    # TODO 暂时使用更低的置信度来卡
    if len(valid_indices) == 0:
        return None, [], []
    # for i, q in enumerate(all_queries):
    #     print("confidence is ", q['confidence'])
    # print("all_queries len is ", len(all_queries))
    # print("valid_indices len is ", len(valid_indices))
    valid_queries = [all_queries[i] for i in valid_indices]

    valid_boxes = torch.stack([q['box_norm'] for q in valid_queries]) # (M,7)
    valid_boxes = decode_box_func(valid_boxes)
    ious = iou3d_nms_utils.boxes_iou3d_gpu(valid_boxes, valid_boxes)
    indep_index =  find_isolated_detections(ious, 0.1)
    # 提取符合条件的查询
    indep_queries = [valid_queries[i] for i in indep_index.cpu().numpy()]
    # 更新 valid_queries：去掉符合条件的查询
    valid_indices = [valid_indices[i] for i in range(len(valid_indices)) if i not in indep_index.cpu().numpy()]
    valid_queries = [valid_queries[i] for i in range(len(valid_queries)) if i not in indep_index.cpu().numpy()]

    if len(valid_indices) == 0:
        return None, [], indep_queries

    center_xy = torch.stack([q['position'] for q in valid_queries]) # (M,2)
    boxes_lw = torch.stack([q['bbox_size'] for q in valid_queries]) # (M,2)
    radius = torch.sqrt(boxes_lw[:,0]**2 + boxes_lw[:,1]**2) / 2.0
    sigma = (radius * 2 + 1) / 6.0 # (M. 1)
    distance = ((center_xy.unsqueeze(1) - center_xy.unsqueeze(0)) ** 2).sum(dim=-1) # (M, M)
    gaussian_mask = (-distance / (2 * sigma[:, None] ** 2 + torch.finfo(torch.float32).eps)).exp()
    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
    attn_mask = gaussian_mask.log()

    # return attn_mask, valid_indices, []
    return attn_mask, valid_indices, indep_queries

def find_isolated_detections(IoUs, threshold=0.1):
    # 获取IoU矩阵的大小
    M = IoUs.shape[0]
    
    # 构造一个掩码，去除对角线上的元素
    # 对角线元素是IoUs[i, i]，我们需要排除这些值，因为它们总是1（与自己重合）
    mask = torch.eye(M, device=IoUs.device).bool()
    
    # 通过广播比较所有IoU值是否都小于阈值
    # 选择IoUs中非对角线的元素（mask为False）
    IoUs_without_diag = IoUs.masked_select(~mask).view(M, M - 1)
    
    # 查找与其他所有框的IoU都小于阈值的索引
    isolated_mask = (IoUs_without_diag < threshold).all(dim=1)  # 对每行进行逻辑“与”操作 (M, M-1) --> (M, )
    
    # 返回符合条件的索引
    isolated_indices = isolated_mask.nonzero(as_tuple=True)[0]
    
    return isolated_indices


def gaussian_attn_mask_from_bboxes_tensor(
    all_box_norms: torch.Tensor,     # (T, 7)
    all_positions: torch.Tensor,     # (T, 2)
    all_sizes: torch.Tensor,         # (T, 2)
    all_confidences: torch.Tensor,   # (T,)
    conf_thresh: float = 0.1,
    decode_box_func=None,
    iou_thresh: float = 0.1
):
    """
    Returns:
      attn_mask     Tensor (Mf, Mf) 或 None
      fuse_orig_idx LongTensor 原始索引, length=Mf
      indep_orig_list Python list 原始索引
    """
    # 1) 过滤低置信度，并保存原始索引映射
    keep_mask = all_confidences >= conf_thresh           # (T,)
    if not keep_mask.any():
        return None, torch.empty(0, dtype=torch.long), []

    orig_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # (M,) 置信度过滤后的原始索引
    box_norms = all_box_norms[keep_mask]                  # (M,7)
    positions = all_positions[keep_mask]                  # (M,2)
    sizes     = all_sizes[keep_mask]                      # (M,2)

    # 2) 解码并计算 IoU
    decoded = decode_box_func(box_norms)                   # (M,7)
    ious = iou3d_nms_utils.boxes_iou3d_gpu(decoded, decoded)  # (M, M)

    # 3) 剔除对角线后找孤立
    M = ious.size(0)
    diag = torch.arange(M, device=ious.device)
    ious[diag, diag] = -1.0                                # 确保 self-iou < iou_thresh
    indep_mask = (ious < iou_thresh).all(dim=1)            # (M,)
    indep_rel = torch.nonzero(indep_mask, as_tuple=False).squeeze(1)  # (K,)

    # 4) 融合部分的相对索引
    fuse_rel = torch.nonzero(~indep_mask, as_tuple=False).squeeze(1)  # (Mf,)

    # 5) 映射回原始 all_box_norms 的索引
    indep_orig = orig_idx[indep_rel].tolist()
    fuse_orig  = orig_idx[fuse_rel]                         # LongTensor (Mf,)

    if fuse_orig.numel() == 0:
        return None, torch.empty(0, dtype=torch.long), indep_orig

    # 6) 计算高斯注意力 Mask（只对要融合的那部分）
    ctr = positions[fuse_rel]                               # (Mf, 2)
    sz  = sizes[fuse_rel]                                   # (Mf, 2)
    radius = torch.norm(sz, dim=1) / 2.0                    # (Mf,)
    sigma  = (radius * 2 + 1) / 6.0                         # (Mf,)

    diff  = ctr.unsqueeze(1) - ctr.unsqueeze(0)             # (Mf, Mf, 2)
    dist2 = (diff**2).sum(-1)                               # (Mf, Mf)

    eps = torch.finfo(torch.float32).eps
    gaussian = torch.exp(-dist2 / (2 * sigma[:,None]**2 + eps))
    gaussian = torch.where(gaussian < eps, torch.zeros_like(gaussian), gaussian)
    attn_mask = torch.log(gaussian + eps)                   # (Mf, Mf)
    
    return attn_mask, fuse_orig, indep_orig




class GroupAttention(nn.Module):
    """
    对一个组内的特征做一次自注意力(Transformer Encoder 的一层简化版本)。
    """
    def __init__(self, d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        """
        x: (B, K, D)
           B: batch_size (可理解为一次处理多个 group 的并行，如果做不到就单组单组算)
           K: 一个组内 Query 数量
           D: 特征维度
        """
        # Self-Attention
        # Q, K, V 全都是 x
        attn_out, _ = self.self_attn(query=x, key=x, value=x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Feed Forward
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x

class Fusion_Decoder(nn.Module):
    """
    对一个组内的特征做一次自注意力(Transformer Encoder 的一层简化版本)。
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=256, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_pos_embed, attn_mask):
        """
        x: (B, K, D)
           B: batch_size (可理解为一次处理多个 group 的并行，如果做不到就单组单组算)
           K: 一个组内 Query 数量
           D: 特征维度
        """
        # Self-Attention 同时也有数据分布跨域适应的作用
        query = self.with_pos_embed(x, x_pos_embed)
        q = k = v = query
        query2, _ = self.self_attn(q, k, v)
        query = query + self.dropout(query2)
        query = self.norm1(query)

        # with attn mask
        attn_out, _ = self.multihead_attn(query, k, v, attn_mask=attn_mask)
        query = query + self.dropout1(attn_out)
        query = self.norm2(query)

        # Feed Forward
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(ff_out)
        query = self.norm3(query)

        return query

class Fusion_Decoder_CDA(nn.Module):
    """
    对一个组内的特征做一次自注意力(Transformer Encoder 的一层简化版本)。
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=256, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_pos_embed, attn_mask):
        """
        x: (B, K, D)
           B: batch_size (可理解为一次处理多个 group 的并行，如果做不到就单组单组算)
           K: 一个组内 Query 数量
           D: 特征维度
        """
        # Self-Attention 同时也有数据分布跨域适应的作用
        query = self.with_pos_embed(x, x_pos_embed)
        q = k = v = query
        query2, _ = self.self_attn(q, k, v)
        query = query + self.dropout(query2)
        query = self.norm1(query)

        # with attn mask
        # attn_out, _ = self.multihead_attn(query, k, v, attn_mask=attn_mask)
        # query = query + self.dropout1(attn_out)
        # query = self.norm2(query)

        # Feed Forward
        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(ff_out)
        query = self.norm3(query)

        return query


#  def get_bboxes(self, preds, align_num=100):
#         '''
#         preds:  # (B_n, query_num, 8)
#         '''
#         out_prob = preds[..., 7:] # (B, 1000, 1) B在验证或者测试的时候一定是 ==1
#         out_bbox = preds[..., :7] # (B, 1000, 7)
#         batch_size = out_prob.shape[0]

#         # out_prob = out_logits.sigmoid() XXX 已经在decoder中计算过sigmoid
#         out_prob = out_prob.view(out_prob.shape[0], -1) # (B, 1000)
#         out_bbox = self.box_decode_func(out_bbox)

#         def _process_output(indices, bboxes):
#             topk_boxes = indices.div(out_prob.shape[2], rounding_mode="floor").unsqueeze(-1)
#             labels = indices % out_prob.shape[2] # 得到标签
#             boxes = torch.gather(bboxes, 0, topk_boxes.repeat(1, out_bbox.shape[-1]))
#             return labels + 1, boxes, topk_boxes

#         new_ret_dict = []
#         all_bboxes = []
#         all_mask = []
#         topk_indices_list = list() # [(n1,), (n2,)...] 筛选对应的索引
#         for i in range(batch_size):
#             out_prob_i = out_prob[i] # （1000*num_class，)
#             out_bbox_i = out_bbox[i] # (1000, 7)

#             topk_indices_i = torch.nonzero(out_prob_i >= 0.2, as_tuple=True)[0] # 筛选置信度大于0.1的的索引 (n, ) TODO 看一下shape
#             scores = out_prob_i[topk_indices_i] # (n, ) 这个因为多cls也是相同的repeat 所以不用上面的操作

#             labels, boxes, topk_indices = _process_output(topk_indices_i.view(-1), out_bbox_i) # 分别得到标签和bbox shape 为 (n, ) and (n, 7)

#             topk_indices_list.append(topk_indices)

#             scores_list = list()
#             labels_list = list()
#             boxes_list = list()
            

#             for c in range(self.num_classes):
#                 mask = (labels - 1) == c # 对于分类无关来说其实是全True ，(n, ), 对于多分类的来说其实就是依次处理每个分类用的
#                 scores_temp = scores[mask]
#                 labels_temp = labels[mask]
#                 boxes_temp = boxes[mask]

#                 scores_list.append(scores_temp)
#                 labels_list.append(labels_temp)
#                 boxes_list.append(boxes_temp)

#             scores = torch.cat(scores_list, dim=0) # (n,)
#             labels = torch.cat(labels_list, dim=0) # (n,) 在类别无关中，其实label是全0
#             boxes = torch.cat(boxes_list, dim=0) # (n,7)
#             # ret = dict(pred_boxes=boxes, pred_scores=scores, pred_labels=labels)
#             # new_ret_dict.append(ret)
#             # 截断或补零
#             boxes = torch.cat([boxes, scores.unsqueeze(-1)], dim=1) # n,8
#             all_bboxes.append(boxes)
#             # n = boxes.size(0)
#             # if n >= align_num:
#             #     aligned_tensor = boxes[:align_num]
#             #     aligned_mask = boxes.new_ones(align_num)  # 全有效
#             # else:
#             #     padding = boxes.new_zeros((align_num - n, 8))
#             #     aligned_tensor = torch.cat([boxes, padding], dim=0)
#             #     aligned_mask = torch.cat([torch.ones(n, dtype=torch.int32), torch.zeros(align_num - n, dtype=torch.int32)])
#             #     aligned_mask = mask.to(boxes)
#             # all_bboxes.append(aligned_tensor)
#             # all_mask.append(aligned_mask.bool())

#         return all_bboxes, topk_indices_list

#     @torch.no_grad()
#     def matcher(self, outputs, targets):

#         pred_logits = outputs["pred_logits"] # (B, 1000, 1)
#         pred_boxes = outputs["pred_boxes"] # (B, 1000, 7)

#         bs, num_queries = pred_logits.shape[:2]
#         # We flatten to compute the cost matrices in a batch
#         out_prob = pred_logits.sigmoid() # (B, 1000, 1)
#         # ([batch_size, num_queries, 6], [batch_size, num_queries, 2])
#         out_bbox = pred_boxes[..., :6]
#         out_rad = pred_boxes[..., 6:7]

#     # Also concat the target labels and boxes
#         # [batch_size, num_target_boxes]
#         tgt_ids = [v["labels"] for v in targets] # [(n1,), (n2,)]
#         # [batch_size, num_target_boxes, 6]
#         tgt_bbox = [v["gt_boxes"][..., :6] for v in targets] # [(n1,6), (n2,6)]
#         # [batch_size, num_target_boxes, 2]
#         tgt_rad = [v["gt_boxes"][..., 6:7] for v in targets] # [(n1,1), (n2,1)]

#         alpha = 0.25
#         gamma = 2.0

#         indices = []

#         for i in range(bs):
#             with torch.cuda.amp.autocast(enabled=False): # 禁用自动混合精度, 强制单精度计算，适合高精度需求场景
#                 out_prob_i = out_prob[i].float()    # (1000, 1)
#                 out_bbox_i = out_bbox[i].float()    # (1000, 6)
#                 out_rad_i = out_rad[i].float()      # (1000, 1)
#                 tgt_bbox_i = tgt_bbox[i].float()    # (n, 6)
#                 tgt_rad_i = tgt_rad[i].float()      # (n, 1)

#                 # [num_queries, num_target_boxes]
#                 cost_giou = -generalized_box3d_iou(
#                     box_cxcyczlwh_to_xyxyxy(out_bbox[i]),
#                     box_cxcyczlwh_to_xyxyxy(tgt_bbox[i]),
#                 ) # (1000, n) 取负数表示GIoU越大，代价越小
#                 # 分类代价计算方式类似Focal Loss，不同的是，这是
#                 neg_cost_class = (1 - alpha) * (out_prob_i ** gamma) * (-(1 - out_prob_i + 1e-8).log()) # (1000, 1) 负样本分类代价 表示得分越高 代价越高
#                 pos_cost_class = alpha * ((1 - out_prob_i) ** gamma) * (-(out_prob_i + 1e-8).log()) # (1000, 1) 正样本代价，得分越高，代价越低
#                 cost_class = pos_cost_class[:, tgt_ids[i]] - neg_cost_class[:, tgt_ids[i]] # 结果shape (1000, n_idx)，tgt_ids为batch中每个样本对应的gt label[(n1,), (n2,)], 在第二维度上筛选，即每个gt都要跟所有的query去计算对应的label损失

#                 # Compute the L1 cost between boxes
#                 # [num_queries, num_target_boxes]
#                 cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1) # p = 1 求的是Manhattan距离，=2为Eucliean距离， 为♾️则是Chebyshev距离
#                 cost_rad = torch.cdist(out_rad_i, tgt_rad_i, p=1)

#             # Final cost matrix
#             C_i = (
#                     self.cost_bbox * cost_bbox
#                     + self.cost_class * cost_class
#                     + self.cost_giou * cost_giou
#                     + self.cost_rad * cost_rad
#             ) # （1000， n）代价矩阵
#             # [num_queries, num_target_boxes]
#             C_i = C_i.view(num_queries, -1).cpu()
#             indice = linear_sum_assignment(C_i) # 匈牙利匹配算法找到最小成本匹配，返回的是一个元组，两个元素都是数组，分别表示最佳匹配的行/列索引
#             indices.append(indice) # 批次结果

#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # 索引数组变张量 由于gt数量远小于object query数量

#     def fuse_features_by_index(self, index_list, feature_list, fusion_func, extra_future, extra_index):
#         """
#         根据索引对特征进行融合。

#         参数:
#         - index_list: list of torch.Tensor, 形状为 (1, n, 1) 的索引张量列表，每个表示有效的索引位置。 eg. [(1,300,1), (1,62,1)...]
#         - feature_list: list of torch.Tensor, 形状为 (1, n, C) 的特征图张量列表。  eg. [(1,300,C), (1,62,C)...]
#         - fusion_func: Callable, 自定义融合函数, 接受输入 (n, k, C)，返回融合后的张量 (n, 1, C),
#                     其中 k 表示参与融合的特征数量。
#         - extra_future: (1, 200, C), ego自身refine了500个query, 其中300个参与融合, 后200个用于从前到后填充不重叠的其他agent的query 
#         - extra_index: (1, 200, 1)

#         返回:
#         - fused_features: torch.Tensor, 融合后的特征张量, 形状为 (1, ego_query_num + extra_query_num, C)。  eg. (1, 300+200, C)
#         """
#         # 检查输入合法性
#         assert len(index_list) == len(feature_list), "索引列表和特征图列表长度不一致"
        
#         # 统一处理索引，获取所有唯一索引
#         all_indices = torch.cat([idx.squeeze(0) for idx in index_list], dim=0)  # (sum(n), 1)
#         # 相同的索引意味着相同的位置, (n_unique, ) 和逆映射 (sum(n),) 表示每个元素在unique_indices中的位置
#         # FIXME 什么情况? 即使设置不用排序，但是最后结果依然排序，想要稳定去重，只能自己写求unique
#         # unique_indices, inverse_indices = torch.unique(all_indices, sorted=False, return_inverse=True) 

#         seen = set()
#         unique_vals = []
#         for val in all_indices:
#             scalar_val = val.item() # 这里debug了好久，tensor对象是不可哈希的，没搞明白直接导致这里去重失败，还会出现重复，因此必须转为python标量
#             if scalar_val not in seen:
#                 seen.add(scalar_val)
#                 unique_vals.append(scalar_val)
#         unique_indices = torch.tensor(unique_vals).to(all_indices)

#         # 构建每个索引对应的特征列表
#         feature_map = {idx.item(): [] for idx in unique_indices} # eg. {id: [(1, C), ...]}
#         for idx, features in zip(index_list, feature_list):
#             for i, ind in enumerate(idx.squeeze(0).squeeze(-1)): # 遍历每个agent的索引
#                 feature_map[ind.item()].append(features[:, i, :])  # 按索引存入特征 (1, C)

#         # 对每个唯一索引进行融合 然后重新放回去 形成{unique_id: [feature]}
#         fused_features = []  # 存储融合后的特征
#         for idx in unique_indices:
#             features_to_fuse = torch.stack(feature_map[idx.item()], dim=1)  # (1, k, C) 同一个空间位置有多个feature, 可能是ego和其他agent，也可能是agent之间
#             fused_features.append(fusion_func(features_to_fuse)) # 融合返回的应该是(1, 1, C)
#         fused_features = torch.cat(fused_features, dim=1)  # (1, n_unique, C)

#         # 从 fused_features 中提取属于 ego 的特征
#         ego_indices = index_list[0].squeeze(0).squeeze(-1)  # ego 的索引 （n1,） ego的索引个数是固定的，就等于query_num
#         ego_mask = torch.isin(unique_indices, ego_indices)  # 找到属于 ego 的索引 (n_unique, ) ego对应的索引就为 True
#         ego_features = fused_features[:, ego_mask, :]  # 提取属于 ego 的部分 (1, ego_query_size, C)

#         non_overlap_features = []
#         for idx, features in zip(index_list[1:], feature_list[1:]): # 忽略 ego
#             mask = ~torch.isin(idx.squeeze(0), index_list[0].squeeze(0)) # 非重叠部分 (n_unique, 1) XXX 首先完全重叠不可能，那只有一种可能，那就是agent和ego感知范围都不重合，所以根本就是空
#             selected_features = features[:, mask.squeeze(), :] # 提取非重叠特征 (1, k', C)
#             if selected_features.size(1) > 0:
#                 non_overlap_features.append(selected_features)

#         # 将非重叠特征按分数截断并填充到最终结果中
#         if len(non_overlap_features) > 0:
#             non_overlap_features = torch.cat(non_overlap_features, dim=1)  # (1, k_all, C)
#             append_num = min(non_overlap_features.size(1), self.extra_query_num) # 最大不超过 extra_query_num
#             extra_future[:, :append_num, :] = non_overlap_features[:,:append_num,:]
#         # else: # 首先能进入融合函数就说明有投影的query存在，结果非重叠的特征是0，这就说明全部是重叠的特征, 经过验证，此时投影过来的特征数量很少，一般是个位数，极少数时候是几十
#         #     print("------------------------------------------------")
#         #     print("Oops! All overlap???")
#         #     print("unique_indices shape is ", unique_indices.shape)
#         #     print("agent 1 shape is ", index_list[1].shape)
#         #     print("------------------------------------------------")

#         # 最终特征: ego + extra_future
#         final_features = torch.cat([ego_features, extra_future], dim=1)  # (1, ego_query_size + etra_query_num, C)

#         unique_indices = unique_indices.unsqueeze(0).unsqueeze(-1) # (1, n_unique, 1)
#         index_num = min(unique_indices.size(1), self.num_queries + self.extra_query_num)
#         assert unique_indices.size(1) >= self.num_queries
#         remain_start = index_num - self.num_queries
#         final_indices = torch.cat([unique_indices[:, :index_num, :], extra_index[:, remain_start:, :]], dim = 1) # 500
#         return final_features, final_indices


# def generalized_box3d_iou(boxes1, boxes2):

#     boxes1 = torch.nan_to_num(boxes1)
#     boxes2 = torch.nan_to_num(boxes2)

#     assert (boxes1[3:] >= boxes1[:3]).all()
#     assert (boxes2[3:] >= boxes2[:3]).all()

#     iou, union = box_iou_wo_angle(boxes1, boxes2)

#     ltb = torch.min(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
#     rbf = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

#     whl = (rbf - ltb).clamp(min=0)  # [N,M,3]
#     vol = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]

#     return iou - (vol - union) / vol # 标准 IoU - (包围盒体积 - 并集体积) / 包围盒体积


from pathlib import Path
import math

def avg_pcd_size_and_log2(dir_path="."):
    """
    统计目录下所有 .pcd 文件平均大小（单位：字节），并给出其 log2。

    Parameters
    ----------
    dir_path : str or Path, default "."
        目标目录路径；支持递归遍历子目录。

    Returns
    -------
    avg_bytes : float
        平均大小（字节）。
    log2_avg  : float
        以 2 为底的平均大小对数。
    """
    pcd_files = list(Path(dir_path).rglob("*.pcd"))  # 递归搜索
    print(pcd_files)

    if not pcd_files:
        raise FileNotFoundError("指定路径下未找到 .pcd 文件")
    total_bytes = sum(f.stat().st_size for f in pcd_files)
    avg_bytes   = total_bytes / len(pcd_files)
    log2_avg    = math.log2(avg_bytes)

    return avg_bytes, log2_avg


# 示例调用
if __name__ == "__main__":
    avg, log2_val = avg_pcd_size_and_log2("/public/home/lilingzhi/xyj/CoAlign/my_dair-v2x/v2x_i/cooperative-vehicle-infrastructure-infrastructure-side-velodyne")
    print(f"平均大小: {avg:.2f} B")
    print(f"log₂(平均大小): {log2_val:.4f}")

