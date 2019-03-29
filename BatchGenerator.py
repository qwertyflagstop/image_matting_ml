import numpy as np
from threading import Thread, Lock
import sys
if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue
from math import *
from keras.utils import Progbar
from PIL import Image

class BatchMaker(object):

    def __init__(self):
        super(BatchMaker, self).__init__()
        self.validation_split = 0.15
        self.epoch_length = 1000*self.validation_split
        self.validation_split_index = 200
        self.augmentation_enabled = False
        self.validation_images = None

    def get_batch(self, offset, batch_size):
        """
            Get a batch synchronously
        :param offset: where in dataset
        :param batch_size: mini batch size
        :return: the X,Y training batch
        """
        X = np.random.normal((batch_size, 64, 64, 3))
        Y = np.zeros((batch_size,100))
        return (X,Y)

    def make_validation_images(self):
        x = []
        y = []
        print('making validation images')
        self.augmentation_enabled = False
        b = Progbar(self.validation_split_index)
        for i in np.arange(0, self.validation_split_index):
            xi,yi = self.get_batch(i,1)
            x.append(xi[0])
            y.append(yi[0])
            b.update(i)
        X = np.array(x)
        Y = np.array(y)
        self.validation_images = (X, Y)
        self.augmentation_enabled = True
        return X,Y


    def make_batch(self, q, offset, size, w_index):
        """
            Subclasses do the magic here to actually create a new mini-batch
            :return: how many batches added to the q
        """
        x, y = self.get_batch(offset, size)
        if x is None or y is None:
            return 0
        q.put((x, y))
        return 1

    def get_validation_batch(self, batch_size):
        X = np.random.normal((batch_size, 64, 64, 3))
        Y = np.zeros((batch_size, 100))
        return (X, Y)


class BatchGenerator(object):
    """
        Class for queueing up batches so models can be trained in parallel
    """
    def __init__(self, q_length, num_procs, batch_maker, batch_size):
        super(BatchGenerator, self).__init__()
        self.q = Queue()
        self.lock = Lock()
        self.offset = 0
        self.num_procs = num_procs
        self.q_length = q_length
        self.batch_maker = batch_maker
        self.batch_size = batch_size
        self.epoch = 0
        self.epoch_progress = 0
        self.bar = Progbar(self.batch_maker.epoch_length)


    def start_queue_runners(self):
        for i in np.arange(0,self.num_procs):
            t = Thread(target=self.__batch_loop, args=(self.q, self.q_length, i))
            t.setDaemon(True)
            t.start()

    def get_next_batch_from_q(self):
        X,Y = self.q.get()
        self.bar.update(self.offset%self.batch_maker.epoch_length)
        return (X,Y)

    def current_epoch(self):
        return np.floor(self.offset/self.batch_maker.epoch_length)

    def num_items_in_q(self):
        return self.q.qsize()

    def __batch_loop(self, q, max_q, worker_num):
        while True:
            s = q.qsize()
            if s < max_q:
                with self.lock:
                    batch_num = self.offset
                    self.offset += self.batch_size
                    self.epoch = int(floor(self.offset/self.batch_maker.epoch_length))
                    self.epoch_progress = self.offset/self.batch_maker.epoch_length
                self.batch_maker.make_batch(q, batch_num, self.batch_size, worker_num)