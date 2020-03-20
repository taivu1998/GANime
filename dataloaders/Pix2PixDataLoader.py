'''
This program implements a dataloader for Pix2Pix.

References:
    https://www.tensorflow.org/tutorials/generative/pix2pix
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os, sys
import numpy as np


from BaseDataLoader import Base_DataLoader

class Pix2Pix_DataLoader(Base_DataLoader):
    ''' A dataloader for Pix2Pix. '''

    def __init__(self, data_path, buffer_size = 400, batch_size = 32,
                 img_width = 256, img_height = 256):
        ''' Initializes the class. '''
        super().__init__()
        self.data_path = data_path
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height

    def load_dataset(self):
        ''' Loads the dataset. '''
        train_dataset = tf.data.Dataset.list_files(os.path.join(self.data_path, 'train/*.jpg'))
        train_dataset = train_dataset.map(self.load_image_train,
                                          num_parallel_calls = tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.buffer_size).batch(self.batch_size)

        test_dataset = tf.data.Dataset.list_files(os.path.join(self.data_path, 'val/*.jpg'))
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(self.batch_size)
        
        return train_dataset, test_dataset
        
    def load_image_train(self, image_file):
        ''' Loads the training set. '''
        input_image, real_image = self.load(image_file, dtype = 'float32')
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def load_image_test(self, image_file):
        ''' Loads the test set. '''
        input_image, real_image = self.load(image_file, dtype = 'float32')
        input_image, real_image = self.resize(input_image, real_image,
                                        self.img_height, self.img_width)
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def load(self, image_file, dtype = 'uint8'):
        ''' Loads an image. '''
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image)

        width = tf.shape(image)[1]
        mid = width // 2

        sketch_image = image[:, mid:, :]
        color_image = image[:, :mid, :]
        
        img_type = eval('tf.' + dtype)
        sketch_image = tf.cast(sketch_image, img_type)
        color_image = tf.cast(color_image, img_type)

        return sketch_image, color_image
        
    def resize(self, input_image, real_image, height, width):
        ''' Resizes two images. '''
        input_image = tf.image.resize(input_image, [height, width],
                                      method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                     method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image
        
    def random_crop(self, input_image, real_image):
        ''' Randomly crops two images. '''
        stacked_image = tf.stack([input_image, real_image], axis = 0)
        cropped_image = tf.image.random_crop(stacked_image,
                                             size = [2, self.img_height, self.img_width, 3])
        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image, real_image):
        ''' Normalizes two images to [-1, 1]. '''
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        ''' Randomly crops and flips two images. '''
        # Resizing to 286 x 286 x 3.
        input_image, real_image = self.resize(input_image, real_image, 286, 286)
        # Randomly cropping to 256 x 256 x 3.
        input_image, real_image = self.random_crop(input_image, real_image)
        if tf.random.uniform(()) > 0.5:
            # Random mirroring.
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
        return input_image, real_image
