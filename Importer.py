import json
from h5py import File, special_dtype
import numpy as np
import os
from Queue import Queue
from threading import Thread, Lock
from pprint import pprint
from io import BytesIO
import matplotlib.pyplot as plt
from keras.utils.generic_utils import Progbar
from PIL import Image

def make_h5py_file(fp, num_entries, classes):
    f = File(fp, 'w')
    f.attrs['classes'] = np.array([x.encode() for x in classes])
    byte_type = special_dtype(vlen=np.dtype('uint8'))
    images_dset = f.create_dataset("images", (num_entries,), dtype=byte_type, maxshape=(None,))
    masks_dset = f.create_dataset("masks", (num_entries,), dtype=byte_type, maxshape=(None,))
    return images_dset, masks_dset, f


def worker(queue, lock):
    global index
    global anno_file_list
    global done_workers
    while True:
        lock.acquire()
        if index>=len(anno_file_list):
            done_workers += 1
            lock.release()
            return
        i = index
        index+=1
        lock.release()
        try:
            mask_path = anno_file_list[i]
            image_path = mask_path.replace('annotations','images').replace('png','jpg')
            jpg_image = Image.open(image_path)
            mask_image = Image.open(mask_path)
            np_image = np.array(jpg_image)
            mask_image = np.array(mask_image)
            if np_image.shape[-1] != 3:
                continue
            string_in = BytesIO()
            jpg_image.save(string_in, format='jpeg')
            string_in.seek(0)
            jpg_image_string = np.fromstring(string_in.read(), dtype='uint8')

            m_image = Image.fromarray(mask_image.astype(np.uint8))
            string_in = BytesIO()
            m_image.save(string_in, format='png')
            string_in.seek(0)
            mask_image_string = np.fromstring(string_in.read(), dtype='uint8')

            queue.put({'image': jpg_image_string, 'mask': mask_image_string})
        except Exception as e:
            print(e)
            continue

class COCOStuffThingsImporter(object):

    def __init__(self, name, train_images_folder, mask_images_folder, class_list_file):
        super(COCOStuffThingsImporter, self).__init__()
        self.name = name
        self.train_dir = train_images_folder
        self.mask_dir = mask_images_folder
        self.class_list = [x.split(' ')[-1].strip() for x in open(class_list_file,'r').readlines()]

    def convert(self):
        global index
        global anno_file_list
        global done_workers
        done_workers=0
        anno_file_list = os.listdir(self.mask_dir)
        anno_file_list = [os.path.join(self.mask_dir,x) for x in anno_file_list]
        index = 0
        num_workers = 5
        queue = Queue()
        lock = Lock()
        worker_count = num_workers
        images_dset, masks_dset, f = make_h5py_file('{}.hdf5'.format(self.name), len(anno_file_list), self.class_list)

        for j in np.arange(0, num_workers):
            t = Thread(target=worker, args=[queue, lock])
            t.daemon = True
            t.start()
        i = 0
        bar = Progbar(len(anno_file_list))
        while done_workers<num_workers:
            try:
                s = queue.get(timeout=10)
                image_bytes = s['image']
                mask_bytes = s['mask']
                images_dset[i] = image_bytes
                masks_dset[i] = mask_bytes
                i += 1
                bar.update(i)
                f.flush()
            except:
                continue #probs timed out
        print('need to resize the dataset to {}'.format(i))
        images_dset.resize(i, axis=0)
        masks_dset.resize(i, axis=0)
        f.close()


if __name__ == '__main__':
    train_or_val = 'val'
    c = COCOStuffThingsImporter(name='coco_val',
                                train_images_folder='dataset/images/{}2017/'.format(train_or_val),
                                mask_images_folder='dataset/annotations/{}2017/'.format(train_or_val),
                                class_list_file='labels.txt')
    c.convert()