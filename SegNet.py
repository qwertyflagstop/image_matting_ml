from Network import Network
from scipy import misc, ndimage
from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, ZeroPadding2D, Lambda, Layer, UpSampling2D, Reshape, Permute
from keras.layers.merge import Concatenate, Add
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.backend import tf as ktf
import numpy as np


class SegNet(Network):
    def defualt_operating_shape(self):
        return (None, None, 3)

    def make_model(self):
        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2

        input = Input((None,None,3))
        x = ZeroPadding2D(padding=(pad, pad))(input)
        x = Conv2D(filter_size, kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((pool_size,pool_size))(x)

        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(128,kernel,padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((pool_size,pool_size))(x)

        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(256,kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((pool_size, pool_size))(x)

        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(512, kernel, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((pool_size, pool_size))(x)

        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(512, kernel, padding='valid')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(size=(pool_size, pool_size))(x)
        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(256, kernel, padding='valid')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(size=(pool_size, pool_size))(x)
        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(128, kernel,  padding='valid')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(size=(pool_size, pool_size))(x)
        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(64, kernel, padding='valid')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(size=(pool_size, pool_size))(x)
        x = ZeroPadding2D(padding=(pad, pad))(x)
        x = Conv2D(32, kernel,  padding='valid')(x)
        x = BatchNormalization()(x)

        x = Conv2D(len(self.classes), 1, border_mode='valid')(x)
        x = Activation('softmax')(x)
        m = Model(inputs=[input], outputs=[x])
        print(m.summary())
        return m



if __name__ == '__main__':
    n = SegNet('segnet',['test' for x in range(0,10)])
