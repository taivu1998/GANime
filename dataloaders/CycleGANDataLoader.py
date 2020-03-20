'''
This program implements a dataloader for CycleGAN.

References:
    https://www.tensorflow.org/tutorials/generative/cyclegan
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import numpy as np


from BaseDataLoader import Base_DataLoader


class CycleGAN_DataLoader(Base_DataLoader):
    ''' A dataloader for CycleGAN. '''
    
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
        train_sketch = tf.data.Dataset.list_files(os.path.join(self.data_path, 'trainA/*.jpg'))
        train_sketch = train_sketch.map(self.load_image_train,
                                        num_parallel_calls = tf.data.experimental.AUTOTUNE)
        train_sketch = train_sketch.cache().shuffle(self.buffer_size).batch(self.batch_size)

        train_color = tf.data.Dataset.list_files(os.path.join(self.data_path, 'trainB/*.jpg'))
        train_color = train_color.map(self.load_image_train,
                                      num_parallel_calls = tf.data.experimental.AUTOTUNE)
        train_color = train_color.cache().shuffle(self.buffer_size).batch(self.batch_size)
            
        test_sketch = tf.data.Dataset.list_files(os.path.join(self.data_path, 'valA/*.jpg'))
        test_sketch = test_sketch.map(self.load_image_test,
                                      num_parallel_calls = tf.data.experimental.AUTOTUNE)
        test_sketch = test_sketch.cache().batch(self.batch_size)

        test_color = tf.data.Dataset.list_files(os.path.join(self.data_path, 'valB/*.jpg'))
        test_color = test_color.map(self.load_image_test,
                                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
        test_color = test_color.cache().batch(self.batch_size)
        
        return train_sketch, train_color, test_sketch, test_color
            
    def load_image_train(self, image_file):
        ''' Loads the training set. '''
        image = self.load(image_file, dtype = 'float32')
        image = self.random_jitter(image)
        image = self.normalize(image)
        return image
        
    def load_image_test(self, image_file):
        ''' Loads the test set. '''
        image = self.load(image_file, dtype = 'float32')
        image = self.resize(image, self.img_height, self.img_width)
        image = self.normalize(image)
        return image

    def load(self, image_file, dtype = 'uint8'):
        ''' Loads an image. '''
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image)
        img_type = eval('tf.' + dtype)
        image = tf.cast(image, img_type)
        return image
        
    def resize(self, image, height, width):
        ''' Resizes an image. '''
        image = tf.image.resize(image, [height, width],
                                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image
        
    def random_crop(self, image):
        ''' Randomly crops an image. '''
        cropped_image = tf.image.random_crop(image,
                                             size = [self.img_height, self.img_width, 3])
        return cropped_image
        
    def normalize(self, image):
        ''' Normalizes an image to [-1, 1]. '''
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image
        
    @tf.function()
    def random_jitter(self, image):
        ''' Randomly crops and flips an image. '''
        # Resizing to 286 x 286 x 3.
        image = self.resize(image, 286, 286)
        # Randomly cropping to 256 x 256 x 3.
        image = self.random_crop(image)
        # Random mirroring.
        image = tf.image.random_flip_left_right(image)
        return image
        
