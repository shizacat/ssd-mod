import torch
import torch.nn as nn
from torchvision.models.resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152)


class ResNet(nn.Module):
    """
    backbone - какую базовую сеть использовать
    backbone_path - путь к претренированному словарю pytorch,
        если не указан то загружается пре тренированная на ImageNet модель

    From the Speed/accuracy trade-offs for modern convolutional
    object detectors paper, the following enhancements were made
    to the backbone:
        - The conv5_x, avgpool, fc and softmax layers were removed
          from the original classification model.
        - All strides in conv4_x are set to 1x1.
    """

    def __init__(self, backbone="resnet50", backbone_path=None):
        super().__init__()
        if backbone == "resnet18":
            backbone = resnet18(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == "resnet34":
            backbone = resnet34(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == "resnet50":
            backbone = resnet50(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == "resnet101":
            backbone = resnet101(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:
            backbone = resnet152(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        r = self.feature_extractor(x)
        return r


class SSD300(nn.Module):
    """
    label_num - количество классов + 1, еще один для фона
    """
    def __init__(self, backbone=ResNet('resnet50'), label_num=81):
        super().__init__()
        # base_network
        self.feature_extractor = backbone

        self.label_num = label_num
        self._build_additional_features(self.feature_extractor.out_channels)
        # Количество боксов на карту признаков (FM)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        # Массивы для фильтров
        # Location. 4ре координаты (dcx, dcy, w, h)
        self.loc = []
        # Вероятность для каждого класса для каждого бокса
        self.conf = []

        # Добавляем фильтры
        items = zip(self.num_defaults, self.feature_extractor.out_channels)
        for nd, oc in items:
            self.loc.append(
                nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1)
            )
            self.conf.append(
                nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1)
            )

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        """Строим дополнительные сверточные слои для получения future map
        Вот ток не понятно почему их по два в каждом
        """
        self.additional_blocks = []

        items = zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])
        for i, (input_size, output_size, channels) in enumerate(items):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels, output_size,
                        kernel_size=3, padding=1, stride=2, bias=False
                    ),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels, output_size,
                        kernel_size=3, bias=False
                    ),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    # kaiming_uniform_ - возможно будет лучше
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        """Shape the classifier to the view of bboxes"""
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((
                l(s).view(s.size(0), 4, -1),
                c(s).view(s.size(0), self.label_num, -1)
            ))

        locs, confs = list(zip(*ret))
        locs = torch.cat(locs, 2).contiguous()
        confs = torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs


# class SSD512(nn.Module):
#     pass

class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')  # reduce=False
        self.dboxes = nn.Parameter(
            dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
            requires_grad=False
        )
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduction='none')  # reduce=None

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()

        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        # print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*(mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret
