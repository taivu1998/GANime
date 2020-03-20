'''
This program implements some helper functions for data pipeline.

References:
    https://www.tensorflow.org/tutorials/generative/style_transfer
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tensor_to_image(tensor):
    ''' Converts a tensor image to a PIL image. '''
    tensor = tensor * 255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def imshow(image, title = None):
    ''' Displays an image. '''
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis = 0)

    plt.imshow(image)
    if title:
        plt.title(title)



    





    


