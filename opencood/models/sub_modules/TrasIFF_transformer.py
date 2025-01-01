import collections
import copy

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import numpy as np
import math

from opencood.models.sub_modules.box_attention import Box3dAttention
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils import box_utils

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
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.m = mom
        self.extra_query_num = 200 # 额外的query数量，用于非重叠位置的补充

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        self.trans_adapter = TransAdapt(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        # self.query_fusion = AttenQueryFusion(d_model)
        # self.ref_fusion = AttenQueryFusion(8)
        self.query_fusion = SimpleGatingFusion()
        self.ref_fusion = BoxGatingFusion()
        self.debug = 0
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)

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

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None, record_len=None, pairwise_t_matrix=None, pairwise_t_matrix_ref=None, box_encode_func=None, box_decode_func=None):
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
        # print("topk_indexes first is ", topk_indexes[0].tolist())
        # print("memory shape is ", memory.shape)
        ego_topk_proposals = topk_proposals[:, :self.num_queries, :] # (B_n, query_num, 8)
        ego_topk_indexes = topk_indexes[:, :self.num_queries, :] # (B_n, query_num, 1)
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

        memory_batch_lst = self.regroup(memory, record_len)
        memory_discrete_batch_lst = self.regroup(memory_discrete, record_len)
        ref_boxes_before_trans_batch_lst = self.regroup(ref_boxes_before_trans, record_len)
        memory_mask_batch_lst = self.regroup(memory_mask, record_len)

        ego_topk_indexes_batch_lst = self.regroup(ego_topk_indexes, record_len)
        extra_topk_indexes_batch_lst = self.regroup(extra_topk_indexes, record_len)
        extra_query_batch_lst = self.regroup(extra_query, record_len)
        extra_topk_proposals_batch_lst = self.regroup(extra_topk_proposals, record_len) #  [(N1, extra_num, 8), (N2, extra_num, 8)...]

        fused_queries = []
        fused_ref_windows = []
        fused_indicies = []
        ego_features = []
        # 将其他的agent的feature 投影到ego坐标系
        for bid in range(len(memory_batch_lst)):
            N = record_len[bid] # number of valid agent
            
            memory_b = memory_batch_lst[bid] # (N, H*W, C) 单独一个样本下的N个agent，其中第一个为ego的feature
            memory_discrete_b = memory_discrete_batch_lst[bid] # (N, C, H, W) Encoder筛选过的留下来，其余全部为空
            ref_boxes_trans_b = ref_boxes_before_trans_batch_lst[bid][:,:7,:,:] # (N, 7, H, W) Encoder筛选过的留下来，其余全部为空
            ref_probs_trans_b = ref_boxes_before_trans_batch_lst[bid][:,7:,:,:] # (N, 1, H, W) Encoder筛选过的留下来，其余全部为空
            memory_mask_b = memory_mask_batch_lst[bid] # (N, 1, H, W)
            t_matrix = pairwise_t_matrix[bid][:N, :N, :, :] # (N, N, 2, 3)
            t_matrix_ref = pairwise_t_matrix_ref[bid][:N, :N, :, :] # (N, N, 4, 4)
            # print("bid is ", bid)
            # print("record_len is ", record_len)
            # print("memory_discrete_b shape is ", memory_discrete_b.shape)
            # print("t_matrix shape is ", t_matrix.shape)
            neighbor_memory = warp_affine_simple(memory_discrete_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, C, H, W)
            ref_boxes_trans_b = warp_affine_simple(ref_boxes_trans_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 7, H, W)
            neighbor_memory_mask = warp_affine_simple(memory_mask_b, t_matrix[0, :, :, :], (H, W), mode='nearest') # (N, 1, H, W)

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
                none_ego_bbox_center = box_decode_func(nef[..., :7].squeeze(0)) # (n, 7) 反归一化

                none_ego_bbox_corner = box_utils.boxes_to_corners_3d(none_ego_bbox_center, 'lwh') # (n, 8, 3)
                projected_none_ego_bbox_corner = box_utils.project_box3d(none_ego_bbox_corner.float(), t_matrix_ref[0,id+1].float())
                projected_none_ego_bbox_center = box_utils.corners_to_boxes_3d(projected_none_ego_bbox_corner, 'lwh') # (n, 7)
                # print("id is ", id)
                # print("t_matrix_ref is ", t_matrix_ref[0,id+1])
                # print("none_ego_bbox_center is ", none_ego_bbox_center)
                # print("projected_none_ego_bbox_center is ", projected_none_ego_bbox_center)
                # xxx
                projected_none_ego_bbox_center = box_encode_func(projected_none_ego_bbox_center) # 重新归一化
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

    def forward(self, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None):
        output = query
        intermediate = []
        intermediate_ref_windows = []
        for idx, layer in enumerate(self.layers):
            if self.cp_flag:
                output = cp.checkpoint(layer, idx, output, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask)
            else:
                output = layer(idx, output, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask)
            new_ref_logits, new_ref_windows = self.detection_head(output, ref_windows[..., :7], idx)
            new_ref_probs = new_ref_logits.sigmoid()
            ref_windows = torch.cat(
                (
                    new_ref_windows.detach(),
                    new_ref_probs.detach(),
                ),
                dim=-1,
            )
            intermediate.append(output)
            intermediate_ref_windows.append(new_ref_windows)
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