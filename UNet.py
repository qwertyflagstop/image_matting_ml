from Network import Network
from scipy import misc, ndimage
from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input,Dropout, Conv2D,  ZeroPadding2D, Lambda, Layer, UpSampling2D, Reshape, Permute
from keras.layers import Concatenate
from keras.layers.merge import Concatenate, Add
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.backend import tf as ktf
import numpy as np


class UNet(Network):
    def defualt_operating_shape(self):
        return (None, None, 3)

    def make_model(self):
        inputs = Input((512,512,3))
        conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, 3,  activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)

        up1 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv3), conv2])
        conv4 = Conv2D(64, 3, activation='relu', padding='same')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, 3,  activation='relu', padding='same')(conv4)

        up2 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv4), conv1])
        conv5 = Conv2D(32, 3,  activation='relu', padding='same')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)

        conv6 = Conv2D(len(self.classes), 1, activation='relu', padding='same')(conv5)

        conv7 = Activation('softmax')(conv6)

        model = Model(inputs=[inputs], outputs=[conv7])

        model.compile(loss="categorical_crossentropy", optimizer=Adam(0.0001), metrics=['accuracy'])
        print(model.summary())
        return model


if __name__ == '__main__':
    n = UNet('UNet',['test' for x in range(0,10)])