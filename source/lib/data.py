"""
Управление набором данных
"""

import os
import bz2
import json
import pickle
import random

import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from lib.utils import calc_iou_tensor, Encoder, dboxes300_coco


class COCODetection(data.Dataset):
    """Для работы с COCO - поиск объекта

    При итерации возвращает кортеж:
        img - изображение как PIL.Image.Image
        img_id - индекс изображения
        (htot, wtot) - размер изображения
        bbox_sizes - отмеченные на изображении боксы (ltrb)
        bbox_labels - метки для боксов

    Metods:
        load/save - Загрузка и сохранения объекта из файла
    """

    def __init__(self, root: str, annotate_file: str, transform=None):
        """
        Args:
            root: Directory with all the images.
            annotate_file: Path to annotaion file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.annotate_file = annotate_file
        self.transform = transform

        self.images = {}
        self.img_keys = []
        self.label_map = {}   # Отображает идентификатор во внутренний индекс
        self.label_info = {}  #

        # Start processing annotation
        with open(self.annotate_file, "r") as fin:
            self.data = json.load(fin)

        # Create labels
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"], img["width"])
            if img_id in self.images:
                raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        # Clear without bboxes
        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())

    @property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def save(self, pklfile):
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx - порядковый индекс в массиве изображений
        """
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.root, fn)
        img = Image.open(img_path).convert("RGB")

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        # for (xc, yc, w, h), bbox_label in img_data[2]:
        for (l, t, w, h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            # l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
            bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels = torch.tensor(bbox_labels)

        if self.transform is not None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels


# Transforms
# ----------
class Compose(transforms.Compose):

    def __call__(self, img, hw: tuple, bbox_sizes, bbox_labels):
        for t in self.transforms:
            img, hw, bbox_sizes, bbox_labels = \
                t(img, hw, bbox_sizes, bbox_labels)
        return img, hw, bbox_sizes, bbox_labels


class Resize:
    """Меняет размер изображения
    Call: img, (htot, wtot), bbox_sizes, bbox_labels
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

        self.trans = transforms.Resize(self.output_size)

    def __call__(self, img, hw: tuple, bbox_sizes, bbox_labels):
        img = self.trans(img)
        new_h = img.height
        new_w = img.width

        return img, (new_h, new_w), bbox_sizes, bbox_labels


class Cropping:
    """ Cropping for SSD, according to original paper
        Choose between following 3 conditions:
        1. Preserve the original image
        2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
        3. Random crop
        Reference to https://github.com/chauhan-utk/src.DomainAdaptation

    Return:
        Возвращает изображение как PIL.Image
        Координаты бокса в системе ltrb
    """
    def __init__(self):
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def __call__(self, img, img_size: tuple, bbox_sizes, bbox_labels):
        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return img, img_size, bbox_sizes, bbox_labels

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            # Implementation use 50 iteration to find possible candidate
            for _ in range(1):
                # suze of each sampled path in [0.1, 1] 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w/h < 0.5 or w/h > 2:
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                ious = calc_iou_tensor(
                    bbox_sizes, torch.tensor([[left, top, right, bottom]])
                )

                # tailor all the bbox_sizes and return
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bbox_sizes whose center not in the cropped image
                xc = 0.5*(bbox_sizes[:, 0] + bbox_sizes[:, 2])
                yc = 0.5*(bbox_sizes[:, 1] + bbox_sizes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bbox_sizes[bbox_sizes[:, 0] < left, 0] = left
                bbox_sizes[bbox_sizes[:, 1] < top, 1] = top
                bbox_sizes[bbox_sizes[:, 2] > right, 2] = right
                bbox_sizes[bbox_sizes[:, 3] > bottom, 3] = bottom

                bbox_sizes = bbox_sizes[masks, :]
                bbox_labels = bbox_labels[masks]

                left_idx = int(left*wtot)
                top_idx = int(top*htot)
                right_idx = int(right*wtot)
                bottom_idx = int(bottom*htot)
                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bbox_sizes[:, 0] = (bbox_sizes[:, 0] - left)/w
                bbox_sizes[:, 1] = (bbox_sizes[:, 1] - top)/h
                bbox_sizes[:, 2] = (bbox_sizes[:, 2] - left)/w
                bbox_sizes[:, 3] = (bbox_sizes[:, 3] - top)/h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx
                return img, (htot, wtot), bbox_sizes, bbox_labels


class RandomHorizontalFlip:
    """Случайное отражение по вертикали с вероятностью p

    Return:
        Возвращает изображение как PIL.Image
        Координаты бокса в системе ltrb
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, hw: tuple, bbox_sizes, bbox_labels):
        if random.random() < self.p:
            x_out = 1.0 - bbox_sizes[:, 2]
            bbox_sizes[:, 2] = 1.0 - bbox_sizes[:, 0]
            bbox_sizes[:, 0] = x_out
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img, hw, bbox_sizes, bbox_labels


class ColorJitter(transforms.ColorJitter):
    def __call__(self, img, hw: tuple, bbox_sizes, bbox_labels):
        img = super().__call__(img)
        return img, hw, bbox_sizes, bbox_labels


class ToTensor(transforms.ToTensor):
    def __call__(self, img, hw: tuple, bbox_sizes, bbox_labels):
        img = super().__call__(img)
        # img = img.contiguous()  # May will be need
        return img, hw, bbox_sizes, bbox_labels


class Normalize(transforms.Normalize):
    def __call__(self, img, hw: tuple, bbox_sizes, bbox_labels):
        img = super().__call__(img)
        return img, hw, bbox_sizes, bbox_labels


class AlignmentBBoxs:
    """Выравниваем размер тэнзора боксов и меток до
    заданного значния
    """

    def __init__(self, max_num: int = 200):
        self.max_num = max_num

    def __call__(self, img, hw: tuple, bbox_sizes, bbox_labels):
        bbox_out = torch.zeros(self.max_num, 4)
        label_out = torch.zeros(self.max_num, dtype=torch.long)
        bbox_out[:bbox_sizes.size(0), :] = bbox_sizes
        label_out[:bbox_labels.size(0)] = bbox_labels
        return img, hw, bbox_out, label_out


class WrapEncoderBoxes:
    """Обертка для кодирования боксов по умолчанию
    в соответствии с теми что на изображении

    Args:
        bboxes - формат ltrb

    Return:
        bboxes - возвращаются в формате xywh
    """
    def __init__(self, dboxes, criteria=0.5):
        self._dboxes = dboxes
        self.criteria = criteria
        self.encoder = Encoder(self._dboxes)

    def __call__(self, img, hw: tuple, bboxes, labels):
        bboxes, labels = self.encoder.encode(
            bboxes, labels, self.criteria
        )
        return img, hw, bboxes, labels


# DataLoaders
# ===========
def get_val_dataloader(data_path, batch_size, num_workers=0, sampler=None):
    dataset = COCODetection(
        os.path.join(data_path, "val2017"),
        os.path.join(data_path, "annotations/instances_val2017.json"),
        Compose([
            Resize((300, 300)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            AlignmentBBoxs()
        ])
    )

    dl = data.DataLoader(
        dataset, batch_size,
        shuffle=False,  # Note: distributed sampler is shuffled :(
        sampler=sampler,
        num_workers=num_workers
    )
    return dl


def get_train_dataloader(data_path, batch_size, num_workers=0, sampler=None):
    dboxes = dboxes300_coco()

    dataset = COCODetection(
        os.path.join(data_path, "train2017"),
        os.path.join(data_path, "annotations/instances_train2017.json"),
        Compose([
            Cropping(),
            # RandomHorizontalFlip(),
            Resize((300, 300)),
            ColorJitter(
                brightness=0.125, contrast=0.5,
                saturation=0.5, hue=0.05
            ),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            WrapEncoderBoxes(dboxes)
        ])
    )

    dl = data.DataLoader(
        dataset, batch_size,
        shuffle=False,  # Note: distributed sampler is shuffled :(
        sampler=sampler,
        num_workers=num_workers
    )
    return dl


def get_val_coco_ground_truth(data_path):
    val_annotate = os.path.join(
        data_path,
        "annotations/instances_val2017.json"
    )
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt
