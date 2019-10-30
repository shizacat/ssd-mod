import itertools
from math import sqrt

import torch
import torch.nn.functional as F
import numpy as np


class DefaultBoxes:

    def __init__(
            self, fig_size: int, feat_size: list, steps: list, scales: list,
            aspect_ratios: list,
            scale_xy=0.1, scale_wh=0.2):
        """Генерирует default boxs для каждой feature map

        Args:
            fig_size - размер входного изображения
            feat_size - список размеров изображений полученных от feature map
            steps - на сколько уменьшается изображение на выходе
                соответствующего feature map.
                Думаю примерно счиатеться (fig_zize/feat_size) с округлением
                к целому
            scales - маштаб, как число поделенное на резмер фигуры для
                соответствующего feature map + 1.
                Минимальный берется за 0.1
                Расчитывается из статьи (paper) как sk
                В примерах для VOC брался диапазон: 0.2-1.05
                COCO min: 0.07; диапазон: 0.15-1.05
            aspect_ratios - список списков соотношения сторон для
                соответствующего feature map.
            scale_xy
            scale_wh
        """
        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

        self.steps = steps
        self.scales = scales

        # ---
        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        # ---
        # содержит: (cx, cy, w-s, h-s)
        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            # 1:1; 1:1 - для sk штрих
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    def __call__(self, order="ltrb") -> torch.Tensor:
        if order == "ltrb":
            return self.dboxes_ltrb
        if order == "xywh":
            return self.dboxes


class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-src
        This function is from https://github.com/kuangliu/pytorch-ssd.
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4),
                     scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4),
                     labels_out (Tensor nboxes),
                     scores_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(0)
        self.nboxes = self.dboxes.size(0)

        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        """
            Args:
                bboxes_in - коордиты в формате ltrb
        """
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        # Transform format to xywh format
        x = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2])
        y = 0.5*(bboxes_out[:, 1] + bboxes_out[:, 3])
        w = -bboxes_out[:, 0] + bboxes_out[:, 2]
        h = -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h

        return bboxes_out, labels_out

    def decode_single(
            self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """Perform non-maximum suppression
        Reference to https://github.com/amdegroot/ssd.pytorch
        """
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []
            # maxdata, maxloc = scores_in.sort()

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i]*len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out = torch.cat(bboxes_out, dim=0)
        labels_out = torch.tensor(labels_out, dtype=torch.long)
        scores_out = torch.cat(scores_out, dim=0)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

    def decode_batch(
            self, bboxes_in, scores_in,  criteria=0.45, max_output=200):
        bboxes, probs = self._scale_back_batch(bboxes_in, scores_in)

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, criteria, max_output))
        return output

    def _scale_back_batch(self, bboxes_in, scores_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = (
            bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] +
            self.dboxes_xywh[:, :, :2]
        )
        bboxes_in[:, :, 2:] = (
            bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]
        )

        # Transform format to ltrb
        l = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2]
        t = bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2]
        b = bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, F.softmax(scores_in, dim=-1)


def dboxes300_coco():
    """Default boxs для набора данных COCO.
    Входное изображение 300x300
    """
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here:
    # https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def calc_iou_tensor(box1: torch.Tensor, box2: torch.Tensor):
    """ Calculation of IoU based on two boxes tensor, each box is [x1,y1,x2,y2]
        Reference to
            - https://github.com/kuangliu/pytorch-src
            - https://github.com/kuangliu/pytorch-ssd.
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] ->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] ->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] ->[N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] ->[N,M,2]
    )

    wh = rb - lt    # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou
