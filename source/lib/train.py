""""""

import math

import torch
from tqdm import tqdm


def train_loop(
        model, loss_func, epoch, optim, train_dataloader, val_dataloader,
        encoder,
        is_cuda=False):
    """Тренировка одной эпохи
    Args:
        model:
        loss_func:
        epoch: Номер выполняемой эпохи
        optim:
    """
    loss_epoch = 0

    batches_count = math.ceil(
        len(train_dataloader.dataset) / train_dataloader.batch_size
    )

    model.train()
    with tqdm(total=batches_count) as progress_bar:
        for nbatch, data in enumerate(train_dataloader):
            imgs = data[0]
            imgs_size = data[2]
            bboxes = data[3]
            labels = data[4]

            ploc, plabel = model(imgs)
            ploc, plabel = ploc.float(), plabel.float()

            gloc, glabel = bboxes, labels
            gloc = gloc.transpose(1, 2).contiguous()

            # ---
            loss = loss_func(ploc, plabel, gloc, glabel)

            loss.backward()
            optim.step()
            optim.zero_grad()

            # Log
            loss_epoch += loss

            progress_bar.update()
            progress_bar.set_description(
                'Epoch {:03d} Loss = {:.5f}'.format(
                    epoch,
                    loss,
                )
            )

    return loss_epoch,


def load_checkpoint(model, checkpoint: str):
    """
    Load model from checkpoint.
    """
    print("Loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]
