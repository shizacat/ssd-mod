#!/usr/bin/env python

"""Тренирует Single Shot MultiBox Detector на COCO

Usage:
    main.py (-d=<data> | --data=<data>)
            [--cuda] [-s=<seed>|--seed=<seed>]
            [--backbone=<backbone>]
            [--backbone-path=<backbone_path>]
            [--batch-size=<batch_size>]
            [--lr=<learning_rate>] [--wd=<weight_decay>]
            [--multistep=<multistep>]
            [-e=<epochs> | --epochs=<epochs>]
            [--checkpoint=<path>] [--save]
            [--mode=<mode>]
    main.py (-h | --help)

Options:
    -h --help
        Показывает это сообщение

    -d <data>, --data <data>
        path to test and training data files

    --cuda
        Использовать доступный GPU

    -s seed, --seed seed
        manually set random seed for torch

    --backbone <backbone>
        Name backbone network:
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        [default: resnet50]

    --backbone-path <backbone_path>
        Path to chekcpointed backbone. It should match the
        backbone model declared with the --backbone argument.
        When it is not provided, pretrained model from torchvision
        will be downloaded.

    -e <epochs>, --epochs <epochs>
        Number of epochs for training. [default: 65]

    --batch-size <batch_size>
        Number of examples for each iteration. [default: 32]

    --lr <learning_rate>
        Learning rate. [default: 2.6e-3]

    --wd <weight_decay>
        Weight decay, momentum argument for SGD optimizer. [default: 0.0005]

    --multistep <multistep>
        List epochs at which to decay learning rate. [default: 43, 54]

    --checkpoint <path>
        Path to model checkpoint file

    --save
        Save model checkpoint

    --mode <mode>
        Режим, в котором будет запущена модель. Может быть:
            training - тренировка модели
            evaluate - оценка точности модели
            create - создание модели и сохранение в файл
        [default: training]
"""

import os
import sys
import math
import time

import torch
from docopt import docopt
import numpy as np
import voluptuous as vol
from torch.utils.tensorboard import SummaryWriter

from lib.utils import dboxes300_coco, Encoder
from lib.data import (
    get_train_dataloader, get_val_dataloader, get_val_coco_ground_truth
)
from lib.model import SSD300, ResNet, Loss
from lib.train import tencent_trick, train_loop, load_checkpoint
from lib.evaluate import evaluate


def save_model(file_path, model, epoch, optimizer, scheduler):
    obj = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        # 'label_map': val_dataset.label_info
    }
    obj['model'] = model.state_dict()
    torch.save(obj, file_path)


def train(args):
    use_cuda = args["--cuda"]

    if args["--seed"] is None:
        args["--seed"] = np.random.randint(1e4)
    else:
        args["--seed"] = int(args["--seed"])

    print("Using seed = {}".format(args["--seed"]))
    torch.manual_seed(args["--seed"])
    np.random.seed(seed=args["--seed"])

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    val_coco_gt = get_val_coco_ground_truth(args["--data"])

    train_dataloader = get_train_dataloader(
        args["--data"],
        batch_size=args["--batch-size"]
    )
    val_dataloader = get_val_dataloader(
        args["--data"],
        batch_size=args["--batch-size"]
    )
    ssd = SSD300(backbone=ResNet(args["--backbone"], args["--backbone-path"]))
    loss_func = Loss(dboxes)
    # args.learning_rate * args.N_gpu * (args.batch_size / 32)
    learning_rate = args["--lr"]  # * (args["--batch-size"] / 32)
    start_epoch = 0

    if use_cuda:
        ssd.cuda()
        loss_func.cuda()

    optimizer = torch.optim.Adam(
        tencent_trick(ssd),
        lr=learning_rate,
        weight_decay=args["--wd"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=args["--multistep"],
        gamma=0.1
    )

    # checkpoint
    if args["--checkpoint"] is not None:
        ch_path = args["--checkpoint"]
        if os.path.isfile(ch_path):
            load_checkpoint(ssd, ch_path)
            checkpoint = torch.load(ch_path)

            start_epoch = checkpoint['epoch']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            sys.exit(1)

    # ---/ Modes
    total_time = 0  # ms

    # Logs
    writer = SummaryWriter(log_dir="./runs")

    # Evaluate
    if args["--mode"] == "evaluate":
        acc = evaluate(ssd, val_dataloader, encoder, val_coco_gt)
        print('Model precision {} mAP'.format(acc))
        return

    # Create
    if args["--mode"] == "create":
        save_model(
            './models/epoch_{}.pt'.format(epoch),
            ssd, epoch, optimizer, scheduler
        )
        return

    # Train
    for epoch in range(start_epoch, args["--epochs"]):
        start_epoch_time = time.time()
        loss_epoch, = train_loop(
            ssd, loss_func,
            epoch,
            optimizer,
            train_dataloader, val_dataloader,
            encoder,
            # iteration,
            # logger
            is_cuda=use_cuda
        )
        scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args["--save"]:
            print("saving model...")
            save_model(
                './models/epoch_{}.pt'.format(epoch),
                ssd, epoch + 1, optimizer, scheduler
            )

        # calculate val precision
        acc = evaluate(ssd, val_dataloader, encoder, val_coco_gt)

        # log
        batches_count = math.ceil(
            len(train_dataloader.dataset) / train_dataloader.batch_size
        )
        loss_avg = loss_epoch / batches_count

        writer.add_scalar("Loss Avg/train", loss_avg, epoch)
        writer.add_scalar("Accuracy/val [mAP]", acc, epoch)
        writer.add_scalar("Time Epoch, ms", end_epoch_time, epoch)

    print('total training time: {}'.format(total_time))


if __name__ == "__main__":
    args = docopt(__doc__)

    schema = vol.Schema({
        "--seed": vol.Any(vol.Coerce(int), None),
        "--backbone": vol.Any(
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            None
        ),
        "--batch-size": vol.Coerce(int),
        "--lr": vol.Coerce(float),
        "--wd": vol.Coerce(float),
        "--epochs": vol.Coerce(int),
    }, extra=vol.ALLOW_EXTRA)

    try:
        args = schema(args)
    except vol.Invalid as ex:
        print("\n".join(["{} - {}".format(e.path, e.msg) for e in ex.errors]))
        sys.exit(-1)

    # Convert multistep
    args["--multistep"] = [int(x) for x in args["--multistep"].split(",")]

    # print(args)
    train(args)
