import collections
import copy

import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

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
    
class FeatureMagnet(nn.Module):
    def __init__(self, d_model=256, nhead=8,  dim_feedforward=1024, dropout=0.1, activation="relu"):
        super.__init__()

        self.query_fusion = MLP(d_model*2, d_model, d_model, 3)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, pos_1d, ego_idx=0):
        '''
        x: [(1, L1, C), (1, L2, C)...] ego和其他agent的query, 这些query已经被CDA模块处理过了
        pos_1d: [(1, L1, 1), (1, L2, 1)...] 标记了在统一坐标系下每个对应query的位置, 这个位置具有唯一性
        ref_windows: [(1, L1, 7), (1, L2, 7)...] # 参考框, 用来BoxAttention
        '''
        B_N = x[0].shape[0]  # Batch size
        final_queries = []  # To hold final queries
        
        # Process each agent and the ego agent separately
        ego_queries = x[ego_idx] # (1, L1, C)
        ego_positions = pos_1d[ego_idx]
        
        # Collect all other agents' queries and positions
        other_agents_queries = x[:ego_idx] + x[ego_idx+1:]
        other_agents_positions = pos_1d[:ego_idx] + pos_1d[ego_idx+1:]
        
        # 1. For the ego agent, we want to collect the queries based on position matching
        ego_position_queries = {}  # Dictionary to hold queries by position
        for i in range(ego_queries.shape[1]):
            pos = ego_positions[0, i].item()  # Get the position (assuming batch_size=1 for simplicity)
            if pos not in ego_position_queries:
                ego_position_queries[pos] = []
            ego_position_queries[pos].append(ego_queries[:, i, :]) # 将(1,C)放入
        
        # 2. For each other agent, we process its queries based on the positions
        for agent_queries, agent_positions in zip(other_agents_queries, other_agents_positions):
            agent_position_queries = {}  # Dictionary to hold queries by position
            for i in range(agent_queries.shape[1]):
                pos = agent_positions[0, i].item()  # Get the position
                if pos not in agent_position_queries:
                    agent_position_queries[pos] = []
                agent_position_queries[pos].append(agent_queries[:, i, :])
            
            # 3. Fuse the agent's queries with the ego's queries at the same position
            for pos, agent_query_list in list(agent_position_queries.items()): # list创建副本 
                if pos in ego_position_queries: # 如果在ego中也有重复的位置 
                    # Fuse the queries using MLP (aggregation of queries at the same position)
                    ego_position_queries[pos] += agent_query_list

                    agent_position_queries.pop(pos)

                    # agent_queries_fused = torch.cat(agent_query_list, dim=1)  # Concatenate queries for the same position
                    ego_queries_fused = torch.cat(ego_position_queries[pos], dim=1)  # Concatenate ego queries for the same position (1, C*2) O为重叠的个数
                    # Perform fusion
                    fused_queries = self.query_fusion(ego_queries_fused)  # MLP fusion (1, C)
                    # Update the ego's query with the fused result
                    ego_position_queries[pos] = [fused_queries]  # Update with the fused queries
        
        # Now combine the final queries for ego and other agents 重新恢复成 (1, l1, C)
        final_ego_queries = torch.cat([val[0] for val in ego_position_queries.values()], dim=0).unsqueeze(0)  # Concatenate all position-based fused queries
        final_queries.append(final_ego_queries)  # Add Ego's final queries

        agent_remain_queries = torch.cat([val[0] for val in agent_position_queries.values()], dim=0).unsqueeze(0) # (1, l2', C)
        final_queries.append(agent_remain_queries)  # Add Ego's final queries

        # Stack all the final queries together
        final_queries = torch.cat(final_queries, dim=1)  # Concatenate queries from ego and agents (1, l1+l2', C) 这个也就是paper中提到的 optimal Q

        k = v =  torch.cat(x, dim=1) # 原始的 所有agent 筛选的query feature 全部concat 在一起，形成(1, l1+l2, C)

        outputs = self.self_attn(final_queries, k, v)
        final_queries = final_queries + self.dropout(outputs)
        final_queries = self.norm1(final_queries)
        outputs = self.linear2(self.dropout(self.activation(self.linear1(final_queries))))
        final_queries = final_queries + self.dropout(outputs)
        final_queries = self.norm2(final_queries)

        return final_queries # (1, L, C) 最终就是输出这个L个query


class CrossDomainAdaption(nn.Module):
    def __init__(self, d_model=256, nhead=8,  dim_feedforward=1024, dropout=0.1, activation="relu"):
        super.__init__()
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
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers)

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
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors)# 返回None，None，(B, 1000, 10)，(B, 1000, 1)

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

        encoder_layer = TransformerEncoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers)

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
            pe = pe.view(B, C, -1).transpose(1, 2)
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]])

        memory = self.encoder(src, src_pos, src_shape, src_start_index, src_anchors)
        query_embed, query_pos, topk_proposals, topk_indexes = self._get_enc_proposals(memory, src_anchors)

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
    def __init__(self, d_model, decoder_layer, num_layers):
        super().__init__()

        self.layers = get_clones(decoder_layer, num_layers)

    def forward(self, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None):
        output = query
        intermediate = []
        intermediate_ref_windows = []
        for idx, layer in enumerate(self.layers):
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