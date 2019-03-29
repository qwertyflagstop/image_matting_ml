from UNet import UNet
from XNet import XNet
from SegmentationBatchMaker import SegnetBatchMaker
from BatchGenerator import BatchGenerator
from h5py import File
import numpy as np
from PIL import Image
from SegmentationBatchMaker import segmentation_to_image
from scipy.ndimage.morphology import binary_fill_holes

if __name__ == '__main__':
    # hdf_file = File('coco_train_people.hdf5')
    # m = SegnetBatchMaker(hdf_file)
    # net = XNet('xnet_people',m.class_list)
    # net.load_weights()
    import os
    dirname = 'Person_Medium_Rotation'
    for f in os.listdir(dirname):