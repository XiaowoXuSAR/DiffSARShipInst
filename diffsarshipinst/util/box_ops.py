# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import math
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2, eps=1e-7):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def distance_box_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=True, CIoU=False, eps=1e-7):
    """在ComputeLoss的__call__函数中调用计算回归损失
    :params box1: gt框  [nums1,4]
    :params box2: 预测框 [nums2,4]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1] + eps

    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1] + eps

    # iou = inter / union
    iou, union = box_iou(box1, box2)  # 200 12

    lt = torch.min(box1[:, None, :2], box2[:, :2])  # 200 12 2
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])  # 200 12 2

    if GIoU or DIoU or CIoU:
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        cw = wh[:, :, 0]  # 两个框的最小闭包区域的width
        ch = wh[:, :, 1]  # 两个框的最小闭包区域的height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            part1 = box1[:, None, :2] - box2[:, :2]
            part2 = box1[:, None, 2:] - box2[:, 2:]
            rho2 = ((part1[:, :, 0] + part1[:, :, 1]) ** 2 + (part2[:, :, 0] + part2[:, :, 1])) / 4

            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47

                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1)[:, None] - torch.atan(w2 / h2), 2)

                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def complete_box_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    """在ComputeLoss的__call__函数中调用计算回归损失
    :params box1: gt框  [nums1,4]
    :params box2: 预测框 [nums2,4]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1] + eps

    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1] + eps

    # iou = inter / union
    iou, union = box_iou(box1, box2)  # 200 12

    lt = torch.min(box1[:, None, :2], box2[:, :2])  # 200 12 2
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])  # 200 12 2

    if GIoU or DIoU or CIoU:
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        cw = wh[:, :, 0]  # 两个框的最小闭包区域的width
        ch = wh[:, :, 1]  # 两个框的最小闭包区域的height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            part1 = box1[:, None, :2] - box2[:, :2]
            part2 = box1[:, None, 2:] - box2[:, 2:]
            rho2 = ((part1[:, :, 0] + part1[:, :, 1]) ** 2 + (part2[:, :, 0] + part2[:, :, 1])) / 4

            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47

                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1)[:, None] - torch.atan(w2 / h2), 2)

                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def alpha_box_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, alphaIoU=True, alp=1, eps=1e-7):
    """在ComputeLoss的__call__函数中调用计算回归损失
    :params box1: gt框  [nums1,4]
    :params box2: 预测框 [nums2,4]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1] + eps

    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1] + eps

    # iou = inter / union
    iou, union = box_iou(box1, box2)  # 200 12

    lt = torch.min(box1[:, None, :2], box2[:, :2])  # 200 12 2
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])  # 200 12 2

    if GIoU or DIoU or alphaIoU:
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        cw = wh[:, :, 0]  # 两个框的最小闭包区域的width
        ch = wh[:, :, 1]  # 两个框的最小闭包区域的height
        if alphaIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            part1 = box1[:, None, :2] - box2[:, :2]
            part2 = box1[:, None, 2:] - box2[:, 2:]
            rho2 = ((part1[:, :, 0] + part1[:, :, 1]) ** 2 + (part2[:, :, 0] + part2[:, :, 1])) / 4

            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif alphaIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47

                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1)[:, None] - torch.atan(w2 / h2), 2)

                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))

                return torch.pow(iou, alp) - (torch.pow(rho2, alp) / torch.pow(c2, alp) + torch.pow((v * alpha), alp))  # alpha-CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def mdp_box_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, MdpIoU=True, eps=1e-7):
    """在ComputeLoss的__call__函数中调用计算回归损失
    :params box1: gt框  [nums1,4]
    :params box2: 预测框 [nums2,4]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1] + eps

    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1] + eps

    # iou = inter / union
    iou, union = box_iou(box1, box2)  # 200 12

    lt = torch.min(box1[:, None, :2], box2[:, :2])  # 200 12 2
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])  # 200 12 2

    if GIoU or DIoU or MdpIoU:
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        cw = wh[:, :, 0]  # 两个框的最小闭包区域的width
        ch = wh[:, :, 1]  # 两个框的最小闭包区域的height
        if MdpIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            part1 = box1[:, None, :2] - box2[:, :2]
            part2 = box1[:, None, 2:] - box2[:, 2:]
            rho2 = ((part1[:, :, 0] + part1[:, :, 1]) ** 2 + (part2[:, :, 0] + part2[:, :, 1])) / 4

            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif MdpIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47

                d1 = (box1[:, None, 0] - box2[:, 0]) ** 2 + (box1[:, None, 1] - box2[:, 1]) ** 2
                d2 = (box1[:, None, 2] - box2[:, 2]) ** 2 + (box1[:, None, 3] - box2[:, 3]) ** 2
                return iou - d1 / (c2 * 2) - d2 / (c2 * 2) # mdpIoU

        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def focused_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, MdpIoU=True, FocalerIoU=True, eps=1e-7, d=0.00, u=0.95):
    """在ComputeLoss的__call__函数中调用计算回归损失
    :params box1: gt框  [nums1,4]
    :params box2: 预测框 [nums2,4]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1] + eps

    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1] + eps

    # iou = inter / union
    iou, union = box_iou(box1, box2)  # 200 12

    lt = torch.min(box1[:, None, :2], box2[:, :2])  # 200 12 2
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])  # 200 12 2

    if FocalerIoU:
        iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00,u=0.95

    if GIoU or DIoU or MdpIoU:
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        cw = wh[:, :, 0]  # 两个框的最小闭包区域的width
        ch = wh[:, :, 1]  # 两个框的最小闭包区域的height
        if MdpIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            part1 = box1[:, None, :2] - box2[:, :2]
            part2 = box1[:, None, 2:] - box2[:, 2:]
            rho2 = ((part1[:, :, 0] + part1[:, :, 1]) ** 2 + (part2[:, :, 0] + part2[:, :, 1])) / 4

            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif MdpIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                d1 = (box1[:, None, 0] - box2[:, 0]) ** 2 + (box1[:, None, 1] - box2[:, 1]) ** 2
                d2 = (box1[:, None, 2] - box2[:, 2]) ** 2 + (box1[:, None, 3] - box2[:, 3]) ** 2
                return iou - d1 / c2 - d2 / c2  # mdpIoU

        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def iou_distance_weight(box1, box2, eps=1e-7):
    """在ComputeLoss的__call__函数中调用计算回归损失
    :params box1: gt框  [nums1,4]
    :params box2: 预测框 [nums2,4]
    :return box1和box2的IoU/GIoU/DIoU/CIoU
    """
    iou, union = box_iou(box1, box2)  # 200 12

    lt = torch.min(box1[:, None, :2], box2[:, :2])  # 200 12 2
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])  # 200 12 2

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    cw = wh[:, :, 0]  # 两个框的最小闭包区域的width
    ch = wh[:, :, 1]  # 两个框的最小闭包区域的height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    part1 = box1[:, None, :2] - box2[:, :2]
    part2 = box1[:, None, 2:] - box2[:, 2:]
    rho2 = ((part1[:, :, 0] + part1[:, :, 1]) ** 2 + (part2[:, :, 0] + part2[:, :, 1])) / 4

    return iou, rho2 / c2
