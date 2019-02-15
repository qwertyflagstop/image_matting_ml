from keras.layers import Conv2D, Lambda, Input, ZeroPadding2D, add, LeakyReLU, BatchNormalization, MaxPooling2D, Layer, concatenate, Activation, SeparableConv2D, GlobalAveragePooling2D
from keras.models import Model, model_from_json
import keras.backend as K
from keras import layers
from keras.applications import ResNet50
from Network import Network

from keras.layers import Concatenate, UpSampling2D
from keras.optimizers import Adam
from h5py import File
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np


def load_model_weights(input_image, last_layer, residuals, weight_path):
    model = Model(input_image, last_layer)
    model.load_weights(weight_path)
    return model, residuals


def xception_base_network(path, input, padding=False):

    #x = Lambda(lambda x: x*2-1)(img_input)
    blocks = []
    x = Conv2D(32, (3, 3), padding='same', use_bias=False, name='block1_conv1')(input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    blocks.insert(0, x)
    x = Conv2D(64, (3, 3),  strides=(2, 2), use_bias=False, name='block1_conv2', padding='same')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    blocks.insert(0,x)
    residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)


    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    blocks.insert(0,x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    blocks.insert(0,x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    blocks.insert(0,x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)
        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    blocks.insert(0, x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    #x = GlobalAveragePooling2D()(x)

    model = Model(input, x, name='xception')

    model.load_weights(path)
    return model, x, blocks

class XNet(Network):

    def __init__(self, name, classes):
        super(XNet, self).__init__(name,classes)

    def make_model(self):
        rgb_input = Input((None, None, 3))
        # depth_input = Input((416,416,1))
        base_model, features, blocks = xception_base_network('weights/xception_basenet.h5', rgb_input)
        x = blocks[0]
        d1 = SeparableConv2D(256, kernel_size=(3, 3), dilation_rate=(1, 1), use_bias=False, padding='same',
                             activation='elu')(x)
        d2 = SeparableConv2D(256, kernel_size=(3, 3), dilation_rate=(2, 2), use_bias=False, padding='same',
                             activation='elu')(x)
        d3 = SeparableConv2D(256, kernel_size=(3, 3), dilation_rate=(3, 3), use_bias=False, padding='same',
                             activation='elu')(x)
        d4 = SeparableConv2D(256, kernel_size=(3, 3), dilation_rate=(4, 4), use_bias=False, padding='same',
                             activation='elu')(x)
        x = add([d1, d2, d3, d4])
        for i in np.arange(1, len(blocks)):
            x = UpSampling2D(size=(2, 2))(x)
            r = SeparableConv2D(2 ** (9 - i), kernel_size=(3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(r)
            x = Activation('relu')(x)
            x = Concatenate(axis=-1)([x, blocks[i]])
            x = SeparableConv2D(2 ** (9 - i), kernel_size=(3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = add([x, r])
        x = Conv2D(len(self.classes), (1, 1), use_bias=True)(x)
        x = Activation('softmax')(x)
        model = Model(inputs=[rgb_input], outputs=[x])
        return model




