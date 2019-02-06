from keras.layers import Input,Conv2D,Softmax,BatchNormalization,ReLU
from keras.models import Model
import numpy as np
import os
import shutil


class Network(object):

    def __init__(self, name, classes):
        super(Network, self).__init__()
        self.name = name
        self.classes = classes
        self.model = self.make_model()

    def defualt_operating_shape(self):
        return (None,None,3)

    def make_model(self):
        i = Input((None,None,3))
        x = Conv2D(10,(3,3), padding='same')(i)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        m = Model(i,x)
        return m

    def load_weights(self):
        save_name = 'weights/{}.h5'.format(self.name)
        if os.path.exists(save_name):
            self.model.load_weights(save_name)
        else:
            print('NO WEIGHTS FOUND FOR {}'.format(self.name))
            exit()

    def save_weights(self):
        if not os.path.exists('weights'):
            os.makedirs('weights')
        save_name = 'weight/{}.h5'.format(self.name)
        self.model.save_weights(save_name)

    def predict(self, x):
        return self.model.predict(x)

    def __call__(self, *args, **kwargs):
        x = args[0]
        if (len(x.shape)==3):
            x = np.expand_dims(x,0)
        y = self.predict(x)
        return y


if __name__ == '__main__':
    print('AYY!')