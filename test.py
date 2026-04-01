'''
This program tests a generative model for line art colorization.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import os, sys
import glob
import random
import numpy as np


sys.path.append('.')
sys.path.append('./data')
sys.path.append('./dataloaders')
sys.path.append('./models')
sys.path.append('./options')
sys.path.append('./utils')


from utils.data_pipeline import *
from options.TestOptions import TestOptions

from dataloaders.NeuralStyleTransferDataLoader import NeuralStyleTransfer_DataLoader
from dataloaders.Pix2PixDataLoader import Pix2Pix_DataLoader
from dataloaders.CycleGANDataLoader import CycleGAN_DataLoader

from models.FastNeuralStyleTransferModel import FastNeuralStyleTransfer
from models.NeuralStyleTransferModel import NeuralStyleTransfer
from models.Pix2PixModel import Pix2Pix
from models.CycleGANModel import CycleGAN
from utils.runtime import configure_seeds, ensure_dir, validate_existing_dir


IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


def parseArgs():
    ''' Reads command line arguments. '''
    args = TestOptions().parse()
    args.output_channels = OUTPUT_CHANNELS
    args.img_width = IMG_WIDTH
    args.img_height = IMG_HEIGHT
    validate_existing_dir(args.data_path, '--data-path')
    return args


def resolve_norm_type(model_name, requested_norm):
    ''' Resolves a model-specific normalization type. '''
    if requested_norm is not None:
        return requested_norm
    if model_name == 'cyclegan':
        return 'instancenorm'
    return 'batchnorm'


def validate_non_empty_paths(path_list, label):
    ''' Validates that a file glob produced at least one path. '''
    if len(path_list) == 0:
        raise ValueError('No images found for {}.'.format(label))


def validate_paired_image_counts(data_path, split_a, split_b):
    ''' Validates that two evaluation directories contain matching image counts. '''
    paths_a = sorted(glob.glob(os.path.join(data_path, split_a, '*.jpg')))
    paths_b = sorted(glob.glob(os.path.join(data_path, split_b, '*.jpg')))

    validate_non_empty_paths(paths_a, os.path.join(data_path, split_a))
    validate_non_empty_paths(paths_b, os.path.join(data_path, split_b))

    if len(paths_a) != len(paths_b):
        raise ValueError('Expected matching image counts in {} and {}, found {} vs {}.'.format(
            os.path.join(data_path, split_a), os.path.join(data_path, split_b),
            len(paths_a), len(paths_b)))
    

def generate_outputs_neural_style_transfer(content_paths, style_paths,
                                           output_path, args):
    ''' Generates output images for Neural Style Transfer. '''
    validate_non_empty_paths(content_paths, os.path.join(args.data_path, 'valA'))
    validate_non_empty_paths(style_paths, os.path.join(args.data_path, 'trainB'))

    for img_type in ['real', 'fake']:
        img_path = os.path.join(output_path, img_type)
        ensure_dir(img_path)
        
    count = 0
    for content_path in content_paths:
        count += 1

        target_name = content_path[content_path.rfind('/') + 1 :]
        target_path = os.path.join(args.data_path, 'valB', target_name)
        target_image = NeuralStyleTransfer_DataLoader(None, None).load(target_path)
        img_name = 'val' + '_img' + str(count).zfill(4) + '_' + 'real' + '.jpg'
        img_path = os.path.join(output_path, 'real', img_name)
        image_jpg = tensor_to_image(target_image / 255)
        image_jpg.save(img_path)
        
        style_path = random.choice(style_paths)
        dataloader = NeuralStyleTransfer_DataLoader(content_path, style_path)
        content_image, style_image = dataloader.load_dataset()
        img_name = 'val' + '_img' + str(count).zfill(4) + '_' + 'fake' + '.jpg'
        img_path = os.path.join(output_path, 'fake', img_name)
        
        if args.model == 'neural_style_transfer':
            model = NeuralStyleTransfer()
            model.build_model()
            model.configure_optimizers()
            model.fit(content_image, style_image, output_path = img_path)
                      
        elif args.model == 'fast_neural_style_transfer':
            model = FastNeuralStyleTransfer()
            model.build_model()
            model.fit(content_image, style_image, output_path = img_path)
            

def generate_outputs_gan(epoch, model, test_dataset, output_path):
    ''' Generates output images for GANs. '''
    epoch_path = os.path.join(output_path, 'Epoch {}'.format(epoch))
    ensure_dir(epoch_path)

    img_paths = {}
    for img_type in ['real', 'fake']:
        img_paths[img_type] = os.path.join(epoch_path, img_type)
        ensure_dir(img_paths[img_type])

    count = 0
    for example_input, example_target in test_dataset:
        example_prediction = model(example_input, training = False)

        real_batch = tf.cast((example_target + 1) * 127.5, tf.uint8)
        fake_batch = tf.cast((example_prediction + 1) * 127.5, tf.uint8)

        batch_size = int(tf.shape(real_batch)[0])
        for batch_index in range(batch_size):
            count += 1
            img_name_real = 'epoch' + str(epoch).zfill(3) + '_' + 'val' + \
                            '_img' + str(count).zfill(4) + '_real.jpg'
            img_name_fake = 'epoch' + str(epoch).zfill(3) + '_' + 'val' + \
                            '_img' + str(count).zfill(4) + '_fake.jpg'
            tf.io.write_file(os.path.join(img_paths['real'], img_name_real),
                             tf.io.encode_jpeg(real_batch[batch_index]))
            tf.io.write_file(os.path.join(img_paths['fake'], img_name_fake),
                             tf.io.encode_jpeg(fake_batch[batch_index]))
        
    
def main():
    ''' Main program. '''
    args = parseArgs()
    configure_seeds(args.seed)
    
    ensure_dir(args.output_path)
    
    if args.model == 'pix2pix':
        dataloader = Pix2Pix_DataLoader(data_path = args.data_path, batch_size = args.batch_size,
                                        img_width = args.img_width, img_height = args.img_height,
                                        augment = False)
        train_dataset, test_dataset = dataloader.load_dataset()
        
        model = Pix2Pix()
        model.build_model(arch_gen = args.arch_gen, arch_disc = args.arch_disc,
                          output_channels = args.output_channels,
                          norm_type = resolve_norm_type(args.model, args.norm))
        model.configure_losses()
        model.configure_optimizers()
        model.configure_checkpoints(checkpoint_path = args.checkpoint_path)
        
        generate_outputs_gan(0, model.generator, test_dataset, output_path = args.output_path)
        for ckpt in model.get_checkpoints():
            model.restore_checkpoint(ckpt)
            epoch = int(model.checkpoint.step)
            generate_outputs_gan(epoch, model.generator, test_dataset, output_path = args.output_path)
    
    elif args.model == 'cyclegan':
        validate_paired_image_counts(args.data_path, 'valA', 'valB')
        dataloader = CycleGAN_DataLoader(data_path = args.data_path, batch_size = args.batch_size,
                                         img_width = args.img_width, img_height = args.img_height,
                                         augment = False)
        train_dataset, _, test_dataset, test_targets = dataloader.load_dataset()
        
        model = CycleGAN()
        model.build_model(arch_gen = args.arch_gen, arch_disc = args.arch_disc,
                          output_channels = args.output_channels,
                          norm_type = resolve_norm_type(args.model, args.norm))
        model.configure_losses()
        model.configure_optimizers()
        model.configure_checkpoints(checkpoint_path = args.checkpoint_path)
        
        generate_outputs_gan(0, model.generator_g,
                             tf.data.Dataset.zip((test_dataset, test_targets)),
                             output_path = args.output_path)
        for ckpt in model.get_checkpoints():
            model.restore_checkpoint(ckpt)
            epoch = int(model.checkpoint.step)
            generate_outputs_gan(epoch, model.generator_g,
                                 tf.data.Dataset.zip((test_dataset, test_targets)),
                                 output_path = args.output_path)
                  
    elif args.model in ['neural_style_transfer', 'fast_neural_style_transfer']:
        content_paths = sorted(glob.glob(os.path.join(args.data_path, 'valA/*.jpg')))
        style_paths = sorted(glob.glob(os.path.join(args.data_path, 'trainB/*.jpg')))
        generate_outputs_neural_style_transfer(content_paths, style_paths,
                                               output_path = args.output_path, args = args)
        

if __name__ == '__main__':
    main()
