"""
Оценка качества модели
"""

import math

import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, encoder, cocoGt, is_cuda=False):
    """
    По сути считает mAP
    """
    model.eval()

    inv_map = {v: k for k, v in dataloader.dataset.label_map.items()}

    ret = np.zeros((0, 7), dtype=np.float32)

    batches_count = math.ceil(
        len(dataloader.dataset) / dataloader.batch_size
    )

    with tqdm(total=batches_count) as progress_bar:
        for nbatch, data in enumerate(dataloader):
            imgs = data[0]
            imgs_id = data[1]
            imgs_size = data[2]
            bboxes = data[3]
            labels = data[4]

            if is_cuda:
                imgs = imgs.cuda()

            # Get predictions
            ploc, plabel = model(imgs)
            ploc, plabel = ploc.float(), plabel.float()

            # Handel batch prediction
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(
                        ploc_i, plabel_i, 0.50, 200
                    )[0]
                except Exception:
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue

                htot, wtot = imgs_size[0][idx], imgs_size[1][idx]
                ret_tmp = []
                for loc_, label_, prob_ in zip(*result):
                    ret_tmp.append([
                        imgs_id[idx],
                        loc_[0] * wtot,
                        loc_[1] * htot,
                        (loc_[2] - loc_[0]) * wtot,
                        (loc_[3] - loc_[1]) * htot,
                        prob_,
                        inv_map[int(label_)]  # Метка из набора данных
                    ])
                ret = np.vstack((
                    ret,
                    np.array(ret_tmp).astype(np.float32)
                ))

            progress_bar.update()

    cocoDt = cocoGt.loadRes(ret)

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()

    E.summarize()
    print("Current AP: {:.5f}".format(E.stats[0]))

    # Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
    return E.stats[0]
