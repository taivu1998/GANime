'''
This program implements a dataloader for Neural Style Transfer.

References:
    https://www.tensorflow.org/tutorials/generative/style_transfer
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import numpy as np


from BaseDataLoader import Base_DataLoader
from utils.image_io import read_image, resolve_tf_dtype


class NeuralStyleTransfer_DataLoader(Base_DataLoader):
    ''' A dataloader for Neural Style Transfer. '''
    
    def __init__(self, content_path, style_path):
        ''' Initializes the class. '''
        super().__init__()
        self.content_path = content_path
        self.style_path = style_path
        
    def load_dataset(self):
        ''' Loads the dataset. '''
        content_image = self.load(self.content_path)
        style_image = self.load(self.style_path)
        return content_image, style_image
        
    def load(self, image_file, dtype = 'uint8'):
        ''' Loads an image. '''
        max_dim = 512
        img = read_image(image_file)
        img_type = resolve_tf_dtype(dtype)
        img = tf.image.convert_image_dtype(img, img_type)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
