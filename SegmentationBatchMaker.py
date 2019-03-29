from BatchGenerator import BatchMaker
from h5py import File
from keras.utils import to_categorical
import io
import numpy as np
from keras.utils.generic_utils import Progbar
from Augment import ImageSegAugmentor
import sys
from PIL import Image

np.set_printoptions(threshold=9999999999)

def normalize_image(image):
    return (255.0 * ((image - image.min()) / (image.max() - image.min()))).astype(np.uint8)

def segmentation_to_image(mask_np, class_list):
    brights = np.ones_like(mask_np) * 255
    brights[np.where(mask_np==0)] = len(class_list)
    mask_hsv = np.stack([(mask_np / float(len(class_list)) * 255), brights, brights], axis=-1).astype(np.uint8)
    im = Image.fromarray(mask_hsv, mode='HSV').convert('RGB')
    return im

class SegnetBatchMaker(BatchMaker):

    def __init__(self, hdf_file):
        super(SegnetBatchMaker, self).__init__()
        self.images_data_set = hdf_file['images']
        self.masks_data_set = hdf_file['masks']
        self.class_list = [x.decode() for x in hdf_file.attrs['classes']]
        self.validation_split_index = 30
        self.epoch_length = self.images_data_set.shape[0]
        self.augmentor = ImageSegAugmentor()
        self.image_size = (640, 640)
        self.min_val_loss = 9999
        self.augmentation_enabled = True
        self.validation_images = None

    def make_validation_images(self):
        x = np.zeros((self.validation_split_index, int(self.image_size[1]), int(self.image_size[0]), 3), dtype=np.float16)
        y = np.zeros((self.validation_split_index, int(self.image_size[1]), int(self.image_size[0]), len(self.class_list)), dtype=np.float16)
        self.augmentation_enabled = False
        indexes = np.arange(self.validation_split_index)
        np.random.shuffle(indexes)
        bar = Progbar(indexes.shape[0])
        i=0
        while i <self.validation_split_index:
            bx,by = self.get_batch(indexes[i], 1)
            if bx is None or by is None:
                print('skiping broken image')
                i+=1
                continue
            x[i] = bx
            y[i] = by
            i += 1
            bar.update(i)
        self.validation_images = (x, y)
        self.augmentation_enabled = True

    def get_batch(self, offset, batch_size, fixed_size=False):
        if self.augmentation_enabled:
            indexes = sorted(np.random.choice(self.images_data_set.shape[0]-1, batch_size, replace=False).tolist())
            b_size = 32 * np.random.randint(10, 15)
        else:
            offset = offset % (self.images_data_set.shape[0]-batch_size)
            indexes = sorted(np.arange(offset, offset+batch_size).tolist())
            b_size = self.image_size[0]
        try:
            image_strings = self.images_data_set[indexes]
            mask_strings = self.masks_data_set[indexes]
            rgb_batch = np.zeros((batch_size,b_size, b_size, 3), dtype=np.float16)
            mask_batch = np.zeros((batch_size, b_size, b_size, len(self.class_list)), dtype=np.float16)
            for i in np.arange(0,batch_size):
                rgb_im = Image.open(io.BytesIO(image_strings[i]))
                small_side = min(rgb_im.size[0], rgb_im.size[1])
                s_length = min(b_size, small_side)
                px = np.random.randint(0, rgb_im.size[0]-s_length+1)
                py = np.random.randint(0, rgb_im.size[1]-s_length+1)
                crop_box =(px,py,px+s_length, py+s_length)
                rgb_im = rgb_im.crop(crop_box)
                rgb_im = rgb_im.resize((b_size,b_size))
                mask_im = Image.open(io.BytesIO(mask_strings[i]))
                mask_im = mask_im.crop(crop_box)
                mask_im = mask_im.resize((b_size,b_size))
                im = np.array(rgb_im).astype(np.uint8)
                ma = (np.array(mask_im)).astype(np.uint8)
                ma[np.where(ma==255)]=0 #255 = 0
                if self.augmentation_enabled:
                   im, ma = self.augmentor.augment_crop_images_masks(im, ma, len(self.class_list))
                im = (im.astype(np.float32) / 255.0)
                ma = to_categorical(ma, num_classes=len(self.class_list)).astype(np.float32)
                rgb_batch[i] = im
                mask_batch[i] = ma
        except Exception as e:
            print(sys.exc_info()[-1].tb_lineno)
            print(e)
            return (None, None)
        return (rgb_batch, mask_batch)

    def sample(self, location, num_samples,name_prefix):
        m = Image.new('RGB', (self.image_size[0]*num_samples, self.image_size[1]*2))
        for j in np.arange(0, num_samples):
            location = np.random.randint(0, self.epoch_length-1)
            rgbs, masks = self.get_batch(location, 1, fixed_size=True)
            im = Image.fromarray(normalize_image(rgbs[0]))
            im_t = segmentation_to_image(np.argmax(masks[0], axis=-1), self.class_list)
            m.paste(im, (j * self.image_size[0], self.image_size[1]* 0))
            m.paste(im_t, (j * self.image_size[0], self.image_size[1]* 1))
        m.save('{}.png'.format(name_prefix))


if __name__ == '__main__':
    hdf_file = File('coco_train_people.hdf5')
    m = SegnetBatchMaker(hdf_file)
    m.augmentation_enabled = True
    m.sample(0,10,'train_sample')
    print('sampled')