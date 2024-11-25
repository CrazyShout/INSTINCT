import torch
from torch.nn import functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc, code_size=7):
    """
    A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its
    detector forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args # 分别是存放gt boxes和label的字典，噪声样本数=3， 标签噪声比例=0.5，box噪声缩放0.4

        known = [(torch.ones_like(t["labels"])).cuda() for t in targets] # 为batch中，每个样本的标签生成一个全 1 的张量, 长度为gt的数量。[(n1,), (n2,)]
        batch_size = len(known)
        known_num = [sum(k) for k in known] # 每个样本有几个gt object [n1, n2]

        unmask_bbox = unmask_label = torch.cat(known) # (n1+n2,) 全1
        labels = torch.cat([t["labels"] for t in targets]) # (n1+n2, )所有的label
        boxes = torch.cat([t["gt_boxes"] for t in targets]) # (n1+n2, 7)所有的boxes
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]) # （n1+n2, ）标记每一个标签的batch id
        # 生成带噪声的标签和边界框
        known_indice = torch.nonzero(unmask_label + unmask_bbox) # 返回非零元素的索引，但是默认返回(n1 + n2, 1)，所以需要view
        known_indice = known_indice.view(-1) # (n1+n2,)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1) # （2*3， n1+n2）--> 2*3*(n1+n2)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1) # (2*3*(n1+n2), )
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1) # (2*3*(n1+n2), )
        known_bboxs = boxes.repeat(2 * dn_number, 1) # （2*3*(n1+n2), 7）
        known_labels_expaned = known_labels.clone() # (2*3*(n1+n2), )
        known_bbox_expand = known_bboxs.clone() # （2*3*(n1+n2), 7）

        if label_noise_ratio > 0 and num_classes > 1: # 标签噪声比例=0.5 我新增了类别无关的筛选，如果类别无关则num_classes==1，那以下代码就是无效的 --xuyunjiang 2024-11-18
            p = torch.rand_like(known_labels_expaned.float()) # (2*3*(n1+n2), ) 随机张量 p，它的每个元素都表示一个标签位置对应的随机概率
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob 小于0.5*0.5 则被选中
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label) # 将旧的label替换成新的
        single_pad = int(max(known_num)) # 所有样本中的gt标签个数取最大值 以下假设n2最大

        pad_size = int(single_pad * 2 * dn_number) # 2 * max_gt_num * 3
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1) # (3, n1+n2) 从0到n1+n2-1
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1) # 生成每组的偏移量 （3, 1） 然后加到每一行的索引上 (3, n1+n2) 
        positive_idx = positive_idx.flatten() # 展平 （3*(n1+n2)， ）
        negative_idx = positive_idx + len(boxes) # 每个位置加上n1+n2，相当于就是往后偏移，（3*(n1+n2)， ）
        if box_noise_scale > 0: # 开始给边界框添加噪声
            known_bbox_ = torch.zeros_like(known_bboxs) # 形状为 (2 * dn_number * (n1 + n2), 7)，存储已知的边界框信息
            # only apply on x and y axis
            known_bbox_[:, :3] = known_bboxs[:, :3] - known_bboxs[:, 3:6] / 2 # 边界框顶点最小坐标
            known_bbox_[:, 3:6] = known_bboxs[:, :3] + known_bboxs[:, 3:6] / 2  # box顶点最大坐标
            known_bbox_[:, 6:] = known_bboxs[:, 6:]

            diff = torch.zeros_like(known_bboxs) # (2 * dn_number * (n1 + n2), 7) 存储每个坐标的扰动范围
            diff[:, :3] = known_bboxs[:, 3:6] / 2 # 尺寸的一半
            diff[:, 3:6] = known_bboxs[:, 3:6] / 2
            diff[:, 6:] = 0.1  # 36 degree

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0 # (2 * dn_number * (n1 + n2), 7) 生成的是0或者1，再乘2-1，则值为-1或者1，表示正负扰动方向
            rand_part = torch.rand_like(known_bboxs) # [0,1) 随机数 (2 * dn_number * (n1 + n2), 7) 
            rand_part[negative_idx] += 1.0 # 负样本扰动更强 额外增加1
            rand_part *= rand_sign # 乘以正负号，形成正负扰动

            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale # 最大扰动范围乘以随机扰动，再乘上噪声缩放强度，最后加到box的坐标上去
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0) # 限制范围
            known_bbox_expand[:, :3] = (known_bbox_[:, :3] + known_bbox_[:, 3:6]) / 2 # 中心位置，位于最大点和最小点连线的中心位置
            known_bbox_expand[:, 3:6] = known_bbox_[:, 3:6] - known_bbox_[:, :3]
            known_bbox_expand[:, 6:] = known_bbox_[:, 6:]

        m = known_labels_expaned.long().to("cuda") #  # (2*3*(n1+n2)，)
        input_label_embed = F.one_hot(m, num_classes=num_classes).float() # 根据类别数，转变成one-hot编码 (2 * dn_number * (n1 + n2), num_classes)
        input_bbox_embed = known_bbox_expand

        padding_label = torch.zeros(pad_size, num_classes).cuda() # (pad_size, num_classes) 其中pad_size = 2 * max_gt_num * 3
        # padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, code_size).cuda() # (pad_size, 7)

        input_query_label = padding_label.repeat(batch_size, 1, 1) # (B, 2 * max_gt_num * 3, num_classes)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) # (B, 2 * max_gt_num * 3, 7)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num): # 其实就是有几个样本，这里也就是batch size的大小
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # like [0,1, 0,1,2] 形状为(n1+n2, ) 其实就是用来后面做mask用的
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()  # (2*dn_number*(n1+n2), )
        if len(known_bid): # known_bid的shape 是 (2*3*(n1+n2), ) 记录每个样本对应的bath id
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed # 将(B, 2 * max_gt_num * 3, num_classes)中赋值上对应的扰动标签
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed # 将(B, 2 * max_gt_num * 3, 7)中赋值上对应的扰动box

        tgt_size = pad_size + num_queries #  2 * max_gt_num * 3 + 1000
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0 # 将所有元素设置为 False  (tgt_size, tgt_size)
        # match query cannot see the reconstruct GTs
        attn_mask[pad_size:, :pad_size] = True
        # gt cannot see queries
        attn_mask[:pad_size, pad_size:] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True # 第 0 组的重建目标不能看到后续的重建目标。
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2] = True # 最后一组的重建目标不能看到之前的重建目标。
            else:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True # 中间组的重建目标不能看到前后的其他重建目标。
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i] = True

        dn_meta = {
            "pad_size": pad_size, # 2 * max_gt_num * 3 
            "num_dn_group": dn_number,
        }
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process_w_ious(outputs_class, outputs_coord, outputs_iou, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    outputs_class: (6, B, pad_size + 1000 + 4*max_gt_num, 3)
    outputs_coord: (6, B, pad_size + 1000 + 4*max_gt_num, 7)
    outputs_iou:   (6, B, pad_size + 1000 + 4*max_gt_num, 1)
    dn_meta: {'pad_size': 2 * max_gt_num * 3, 'num_dn_group': 3}
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :] # (6, B, pad_size, 1)
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :] # (6, B, pad_size, 7)
        output_known_iou = outputs_iou[:, :, : dn_meta["pad_size"], :]     # (6, B, pad_size, 1)
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]      # (6, B, 1000 + 4*max_gt_num, 1)
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]      # (6, B, 1000 + 4*max_gt_num, 7)
        outputs_iou = outputs_iou[:, :, dn_meta["pad_size"] :, :]          # (6, B, 1000 + 4*max_gt_num, 1)
        out = {
            "pred_logits": output_known_class[-1],                         # (B, pad_size, 3) 最后一层的预测结果 这个是噪声样本
            "pred_boxes": output_known_coord[-1],
            "pred_ious": output_known_iou[-1]
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class[:-1], output_known_coord[:-1], output_known_iou[:-1]) # 去噪需要辅助损失，拆成一个五个元素的列表，每个元素为字典 
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord, outputs_iou


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]
        out = {
            "pred_logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class[:-1], output_known_coord[:-1])
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord