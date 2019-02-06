#!/usr/bin/env bash
# Download everything
# https://github.com/nightrome/cocostuff
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
wget https://raw.githubusercontent.com/nightrome/cocostuff/master/labels.txt
# Unpack everything
mkdir -p dataset/images
mkdir -p dataset/annotations
unzip train2017.zip -d dataset/images/
unzip val2017.zip -d dataset/images/
unzip stuffthingmaps_trainval2017.zip -d dataset/annotations/