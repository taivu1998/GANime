'''
This program implements a Fast Neural Style Transfer model.

References:
    https://www.tensorflow.org/tutorials/generative/style_transfer
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub

import os, sys
import time
import numpy as np


from BaseModel import BaseModel
from utils.data_pipeline import *


class FastNeuralStyleTransfer(BaseModel):
    ''' A Fast Neural Style Transfer model. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        super().__init__()
        
    def build_model(self):
        ''' Builds network architectures. '''
        hub_module_path = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1'
        self.hub_module = hub.load(hub_module_path)
    
    def fit(self, content_image, style_image, output_path = 'stylized_image_fast.png'):
        ''' Trains the model. '''
        start = time.time()
        self.stylized_image = self.hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        self.save_output(output_path)
        end = time.time()
        print("Total time: {:.1f}".format(end - start))
        
    def predict(self):
        ''' Generates an output image from an input. '''
        return self.stylized_image
    
    def save_output(self, img_path):
        ''' Saves the output image. '''
        output = tensor_to_image(self.stylized_image)
        output.save(img_path)
