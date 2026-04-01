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

try:
    from utils.image_io import read_image, write_image
    from utils.runtime import ensure_dir
except ImportError:
    from image_io import read_image, write_image
    from runtime import ensure_dir


IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    ''' Loads an image. '''
    image = read_image(image_file)

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
    write_image(image, save_path, image_format = 'jpeg')


def preprocess_dataset_pix2pix(data_path, data_preprocessed_path):
    ''' Preprocesses data for Pix2Pix. '''
    if not os.path.isdir(data_path):
        raise ValueError('Source dataset directory does not exist: {}'.format(data_path))

    for split in ['train', 'val']:
        ensure_dir(os.path.join(data_preprocessed_path, split))
        image_folder_path = os.path.join(data_path, split)
        if not os.path.isdir(image_folder_path):
            raise ValueError('Expected split directory does not exist: {}'.format(image_folder_path))
        for image_path in glob.glob(image_folder_path  + '/*.png'):
            img_name = image_path[image_path.rfind('/') + 1 : image_path.rfind('.')]
            save_path = os.path.join(data_preprocessed_path, split, img_name + '.jpg')
            preprocess_pix2pix(image_path, save_path)


def preprocess_cyclegan(image_path, save_path_sketch, save_path_color):
    ''' Preprocesses an example for CycleGAN. '''
    sketch_image, color_image = load(image_path)
    sketch_image, color_image = resize(sketch_image, color_image, IMG_HEIGHT, IMG_WIDTH)
    write_image(sketch_image, save_path_sketch, image_format = 'jpeg')
    write_image(color_image, save_path_color, image_format = 'jpeg')
    
    
def preprocess_dataset_cyclegan(data_path, data_preprocessed_path):
    ''' Preprocesses data for CycleGAN. '''
    if not os.path.isdir(data_path):
        raise ValueError('Source dataset directory does not exist: {}'.format(data_path))

    for split in ['train', 'val']:
        ensure_dir(os.path.join(data_preprocessed_path, split + 'A'))
        ensure_dir(os.path.join(data_preprocessed_path, split + 'B'))
        image_folder_path = os.path.join(data_path, split)
        if not os.path.isdir(image_folder_path):
            raise ValueError('Expected split directory does not exist: {}'.format(image_folder_path))
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
    parser.add_argument('--data-path', type = str,
                        default = os.path.join(os.getcwd(), 'data', 'anime-sketch-colorization-pair'),
                        help = 'Path to the source dataset with PNG image pairs.')
    parser.add_argument('--output-path', type = str,
                        default = os.path.join(os.getcwd(), 'data', 'anime-sketch-colorization-pair-resized'),
                        help = 'Path to the preprocessed output dataset.')
    args = parser.parse_args()
    return args
    
    
def main():
    ''' Main program. '''
    args = parseArgs()
    if args.model == 'cyclegan':
        preprocess_dataset_cyclegan(args.data_path, args.output_path)
    else:
        preprocess_dataset_pix2pix(args.data_path, args.output_path)
    
    
if __name__ == '__main__':
    main()
