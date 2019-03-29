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
    hdf_file = File('coco_train_people.hdf5')
    m = SegnetBatchMaker(hdf_file)
    net = XNet('xnet_people',m.class_list)
    net.load_weights()
    import os
    for f in os.listdir('Person_Medium_Rotation/Person_Medium_Rotation_TIFF'):
        img = Image.open('{}/{}'.format('Person_Medium_Rotation/Person_Medium_Rotation_TIFF',f))
        w,h = img.size
        w_pad = (32 - (w % 32))
        h_pad = (32 - (h % 32))
        w = w_pad + w
        h = h_pad + h
        p_img = Image.new('RGB',(w,h))
        p_img.paste(img,(w_pad/2, h_pad/2))
        # small_side = min(img.size[0], img.size[1])
        # s_length = small_side
        # s_length = int(32 * round(float(s_length) / 32)) #round to multiple of 32
        # px = img.size[0]*0.5 - s_length*0.5
        # py = img.size[1]*0.5 - s_length*0.5
        # crop_box = (px, py, px + s_length, py + s_length)
        # img = img.crop(crop_box)
        # img = img.resize((s_length, s_length))
        # img.save('samples/'+f.replace('jpg', '')+'_input.png')
        n_img = np.array(p_img) /255.0
        if (len(n_img.shape) < 4):
            n_img = np.expand_dims(n_img, axis=0)
        samp = net.model.predict(n_img)
        numpy_image = np.squeeze(np.argmax(samp, axis=-1).astype(np.uint8))
        numpy_image = binary_fill_holes(numpy_image).astype(np.uint8)
        #numpy_image = binary_fill_holes(numpy_image).astype(np.uint8)
        seg_image = segmentation_to_image(numpy_image,m.class_list)
        crop_box = (w_pad/2, h_pad/2, p_img.size[0]-w_pad/2, p_img.size[1]-h_pad/2)
        seg_image = seg_image.crop(crop_box)
        seg_image.save('samples/'+f.replace('tif','png'))
        print('Saved {}x{} to {}'.format(seg_image.size[0],seg_image.size[1],'samples/'+f.replace('tif','png')))

