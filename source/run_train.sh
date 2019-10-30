#!/bin/bash

./main.py -d ../coco \
	--backbone resnet18 \
	-e 10 \
	--save \
	--batch-size 32 \
	--lr 2.6e-3 \
	--mode training

	# --backbone-path \
	# --checkpoint \
