from keras.layers import Input,Conv2D,Softmax,BatchNormalization,ReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np
import os
import shutil
from PIL import Image
from Logger import Logger
from keras.utils.vis_utils import plot_model



def normalize_image(image):
    return (255.0 * ((image - image.min()) / (image.max() - image.min()))).astype(np.uint8)

def segmentation_to_image(mask_np, class_list):
    brights = np.ones_like(mask_np) *255
    brights[np.where(mask_np==0)] = 0
    mask_hsv = np.stack([(mask_np / float(len(class_list)) * 255), brights, brights], axis=-1).astype(np.uint8)
    im = Image.fromarray(mask_hsv,mode='HSV').convert('RGB')
    return im


class Network(object):

    def __init__(self, name, classes):
        super(Network, self).__init__()
        self.name = name
        self.classes = classes
        self.model = self.make_model()
        self.logger = Logger()
        self.batches = 0
        self.min_loss = 99999

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
        save_name = 'weights/{}.h5'.format(self.name)
        self.model.save_weights(save_name)

    def predict(self, x):
        return self.model.predict(x)

    def __call__(self, *args, **kwargs):
        x = args[0]
        if (len(x.shape)==3):
            x = np.expand_dims(x,0)
        y = self.predict(x)
        return y

    def train_with_batch_generator(self, generator):
        trainable_model = multi_gpu_model(self.model, 2, cpu_merge=True)
        trainable_model.compile(Adam(0.001), loss='categorical_crossentropy')
        plot_model(self.model, '{}_model_diagram.png'.format(self.name), show_shapes=True,show_layer_names=True)
        while True:
            X, Y = generator.get_next_batch_from_q()
            loss = trainable_model.train_on_batch(X, Y)
            self.logger.log_scalar('Loss', value=loss, step=self.batches)
            if self.batches%500==0:
                generator.batch_maker.augmentation_enabled = False
                X, Y = generator.batch_maker.validation_images
                total_l = trainable_model.evaluate(X, Y, batch_size=generator.batch_size)
                if total_l < self.min_loss:
                    self.min_loss = total_l
                    self.save_weights()
                self.logger.log_scalar('Val_loss', total_l, self.batches)
                x, y = generator.batch_maker.validation_images
                for i in np.arange(0, 10):
                    self.logger.log_image('val_images_epoch_{}'.format(self.batches), x[i], step=i * 3 + 0)
                    numpy_image = np.squeeze(np.argmax(y[i:i + 1], axis=-1).astype(np.float32) / len(self.classes), 0)
                    self.logger.log_image('val_images_epoch_{}'.format(self.batches), numpy_image, step=i * 3 + 1)
                    samp = self.model.predict(x[i:i + 1])
                    numpy_image = np.squeeze(np.argmax(samp, axis=-1).astype(np.float32) / len(self.classes), 0)
                    self.logger.log_image('val_images_epoch_{}'.format(self.batches), numpy_image, step=i * 3 + 2)
                generator.batch_maker.augmentation_enabled = True
            self.batches += 1


    def sample(self, image, truth=None, name=None):
        if (len(image.shape) < 4):
            image = np.expand_dims(image, axis=0)
        if (len(truth.shape) < 4):
            truth = np.expand_dims(truth, axis=0)
        samp = self.model.predict(image)
        numpy_image = np.squeeze(np.argmax(samp, axis=-1).astype(np.float32) / len(self.classes) * 255.0, 0).astype(
            np.uint8)
        numpy_truth = np.squeeze(np.argmax(truth, axis=-1).astype(np.float32) / len(self.classes) * 255.0, 0).astype(
            np.uint8)
        im = Image.fromarray(numpy_image).convert('RGB')
        im_t = Image.fromarray(numpy_truth).convert('RGB')
        rgb_dat = np.squeeze(normalize_image(image), axis=0)
        im_rgb = Image.fromarray(rgb_dat)
        if name is not None:
            im.save('{}.png'.format(name))
            im_t.save('{}_t.png'.format(name))
            im_rgb.save('{}_rgb.png'.format(name))
        return im, im_t, im_rgb


if __name__ == '__main__':
    print('AYY!')