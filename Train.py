from UNet import UNet
from SegNet import SegNet
from SegmentationBatchMaker import SegnetBatchMaker
from BatchGenerator import BatchGenerator
from h5py import File
from XNet import XNet

if __name__ == '__main__':
    hdf_file = File('coco_train_people.hdf5')
    m = SegnetBatchMaker(hdf_file)
    net = XNet('xnet_people',m.class_list)
    net.load_weights()
    m.make_validation_images()
    m.augmentation_enabled = True
    generator = BatchGenerator(5,6,m,8)
    generator.start_queue_runners()
    net.train_with_batch_generator(generator)