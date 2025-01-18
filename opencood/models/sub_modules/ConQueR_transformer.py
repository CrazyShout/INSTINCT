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

        # H, W = src_shape[0,0], src_shape[0,1]
        # memory_vis = torch.zeros_like(memory).to(memory) # (B, HW, C)
        # fined_query = torch.gather(memory, 1, topk_indexes.expand(-1, -1, memory.shape[-1]))
        # memory_vis = memory_vis.scatter(1, topk_indexes.repeat(1, 1, memory_vis.size(-1)), fined_query)
        # memory_copy = copy.deepcopy(memory)
        # memory_copy = memory_copy.permute(0, 2, 1).reshape(memory.shape[0], 256, H, W)
        # memory_vis = memory_vis.permute(0, 2, 1).reshape(memory_vis.shape[0], 256, H, W)

        # import matplotlib.pyplot as plt
        # import os
        # if self.sample_idx % 20 == 0:
        #     save_dir = "./feature_vis_res_300query"
        #     os.makedirs(save_dir, exist_ok=True)
        #     for b in range(B):
        #         # confidence = memory[b, 7, :, :] # (H, W)
        #         # mask = (confidence > 0.25).float()
        #         # mask = mask.unsqueeze(1)
        #         feature_map = memory_copy[b] # 
        #         feature_map = feature_map.mean(dim=0) # (H, W)
        #         feature_discrete = memory_vis[b]
        #         feature_discrete = feature_discrete.mean(dim=0) # (H, W)

        #         # feature_mask = mask

        #         # 将特征图归一化到 [0, 255]
        #         def normalize_to_image(tensor):
        #             tensor = tensor - tensor.min()
        #             tensor = tensor / tensor.max()
        #             return (tensor * 255).byte()
                
        #         dense_feature = normalize_to_image(feature_map)
        #         feature_discrete = normalize_to_image(feature_discrete)
        #         # 转为 NumPy 格式
        #         dense_feature_np = dense_feature.cpu().numpy()
        #         feature_mask_np = feature_discrete.cpu().numpy()

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
        #         plt.savefig(os.path.join(save_dir, f"feature_map_{self.sample_idx}_{b}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        #         plt.close() 
        # self.sample_idx += 1

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