import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.python.summary import summary

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir='logs'):
        """Creates a summary writer logging to log_dir."""
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print('removed log dir')
        os.makedirs(log_dir)
        self.writer = tf.summary.FileWriter(log_dir)
        self.writer_1 = tf.summary.FileWriter(log_dir+'/scalar1')
        self.writer_2 = tf.summary.FileWriter(log_dir+'/scalar2')

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_scalars(self, tag, value_1, value_2, step):
        """
            useful for comparing values
        :param tag: name
        :param value_1: val 1
        :param value_2: val 2
        :param step: current step
        :return: lol nothing
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value_1)])
        self.writer_1.add_summary(summary, step)
        self.writer_1.flush()
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value_2)])
        self.writer_2.add_summary(summary, step)
        self.writer_2.flush()

    def log_image(self, tag, image, step):
        im_summaries = []
        rescaled = 255.0 * ((image - image.min()) / (image.max() - image.min()))
        im = Image.fromarray(rescaled.astype(np.uint8))
        s = BytesIO()
        plt.imsave(s, im, format='png')

        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=image.shape[0], width=image.shape[1])
        im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, step), image=img_sum))

        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_graph(self, graph, graph_def):
            self.writer.add_graph(graph)
