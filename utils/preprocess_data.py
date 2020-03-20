'''
This program implements data preprocessing.

References:
    https://www.tensorflow.org/tutorials/generative/pix2pix
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import glob
import argparse
import numpy as np


IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    ''' Loads an image. '''
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)

    width = tf.shape(image)[1]
    mid = width // 2
    
    sketch_image = image[:, mid:, :]
    color_image = image[:, :mid, :]
    
    return sketch_image, color_image


def resize(input_image, real_image, height, width):
    ''' Resizes two images. '''
    input_image = tf.image.resize(input_image, [height, width],
                                  method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def preprocess_pix2pix(image_path, save_path):
    ''' Preprocesses an example for Pix2Pix. '''
    sketch_image, color_image = load(image_path)
    sketch_image, color_image = resize(sketch_image, color_image, IMG_HEIGHT, IMG_WIDTH)
    image = tf.concat([color_image, sketch_image], axis = 1)
    image_jpg = tf.io.encode_jpeg(image)
    tf.io.write_file(save_path, image_jpg)


def preprocess_dataset_pix2pix():
    ''' Preprocesses data for Pix2Pix. '''
    data_path = os.path.join(os.getcwd(), 'data',
                             'anime-sketch-colorization-pair')
    data_preprocessed_path = os.path.join(os.getcwd(), 'data',
                                          'anime-sketch-colorization-pair-resized')
                                          
    for split in ['train', 'val']:
        image_folder_path = os.path.join(data_path, split)
        for image_path in glob.glob(image_folder_path  + '/*.png'):
            img_name = image_path[image_path.rfind('/') + 1 : image_path.rfind('.')]
            save_path = os.path.join(data_preprocessed_path, split, img_name + '.jpg')
            preprocess_pix2pix(image_path, save_path)


def preprocess_cyclegan(image_path, save_path_sketch, save_path_color):
    ''' Preprocesses an example for CycleGAN. '''
    sketch_image, color_image = load(image_path)
    sketch_image, color_image = resize(sketch_image, color_image, IMG_HEIGHT, IMG_WIDTH)
    sketch_jpg = tf.io.encode_jpeg(sketch_image)
    color_jpg = tf.io.encode_jpeg(color_image)
    tf.io.write_file(save_path_sketch, sketch_jpg)
    tf.io.write_file(save_path_color, color_jpg)
    
    
def preprocess_dataset_cyclegan():
    ''' Preprocesses data for CycleGAN. '''
    data_path = os.path.join(os.getcwd(), 'data',
                             'anime-sketch-colorization-pair')
    data_preprocessed_path = os.path.join(os.getcwd(), 'data',
                                          'anime-sketch-colorization-pair-resized')
                                          
    for split in ['train', 'val']:
        image_folder_path = os.path.join(data_path, split)
        for image_path in glob.glob(image_folder_path  + '/*.png'):
            img_name = image_path[image_path.rfind('/') + 1 : image_path.rfind('.')]
            save_path_sketch = os.path.join(data_preprocessed_path, split + 'A', img_name + '.jpg')
            save_path_color = os.path.join(data_preprocessed_path, split + 'B', img_name + '.jpg')
            preprocess_cyclegan(image_path, save_path_sketch, save_path_color)


def parseArgs():
    ''' Reads command line arguments. '''
    model_options = ['neural_style_transfer', 'fast_neural_style_transfer',
                     'pix2pix', 'cyclegan']
    parser = argparse.ArgumentParser(description = 'PyTorch ResNet Training.',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type = str, default = 'pix2pix',
                        help = 'Model.', choices = model_options)
    args = parser.parse_known_args()[0]
    return args
    
    
def main():
    ''' Main program. '''
    args = parseArgs()
    if args.model == 'cyclegan':
        preprocess_dataset_cyclegan()
    else:
        preprocess_dataset_pix2pix()
    
    
if __name__ == '__main__':
    main()
