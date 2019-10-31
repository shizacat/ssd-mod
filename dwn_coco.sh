#!/bin/bash

# Downloader dataset coco

mkdir -p coco

(
    cd coco

    if [ ! -d "train2017" ]; then
        wget http://images.cocodataset.org/zips/train2017.zip
        unzip train2017.zip
        rm -rf train2017.zip
    fi

    if [ ! -d "val2017" ]; then
        wget http://images.cocodataset.org/zips/val2017.zip
        unzip val2017.zip
        rm -rf val2017.zip
    fi

    echo "Download Annotation"
    if [ ! -d "annotations" ]; then
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip annotations_trainval2017.zip
        rm -rf annotations_trainval2017.zip
    fi

    echo "Download Test"
    if [ ! -d "test2017" ]; then
        wget http://images.cocodataset.org/zips/test2017.zip
        unzip test2017.zip
        rm -rf test2017.zip
    fi
)
