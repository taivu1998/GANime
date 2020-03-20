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


IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


def parseArgs():
    ''' Reads command line arguments. '''
    args = TestOptions().parse()
    args.output_channels = OUTPUT_CHANNELS
    args.img_width = IMG_WIDTH
    args.img_height = IMG_HEIGHT
    return args
    

def generate_outputs_neural_style_transfer(content_paths, style_paths,
                                           output_path, args):
    ''' Generates output images for Neural Style Transfer. '''
    for img_type in ['real', 'fake']:
        img_path = os.path.join(output_path, img_type)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        
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
    if not os.path.isdir(epoch_path):
        os.mkdir(epoch_path)

    img_paths = {}
    for img_type in ['real', 'fake']:
        img_paths[img_type] = os.path.join(epoch_path, img_type)
        if not os.path.isdir(img_paths[img_type]):
            os.mkdir(img_paths[img_type])

    count = 0
    for n, (example_input, example_target) in test_dataset.enumerate():
        count += 1
        example_prediction = model(example_input, training = True)
        img_list = [example_target[0], example_prediction[0]]
        img_list = [tf.cast((img + 1) * 127.5, tf.uint8) for img in img_list]
        img_type = ['real', 'fake']

        for i in range(2):
            img_name = 'epoch' + str(epoch).zfill(3) + '_' + 'val' + \
                        '_img' + str(count).zfill(4) + '_' + img_type[i] + '.jpg'
            img_path = os.path.join(img_paths[img_type[i]], img_name)
            image_jpg = tf.io.encode_jpeg(img_list[i])
            tf.io.write_file(img_path, image_jpg)
        
    
def main():
    ''' Main program. '''
    args = parseArgs()
    
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)
    
    if args.model == 'pix2pix':
        dataloader = Pix2Pix_DataLoader(data_path = args.data_path, batch_size = args.batch_size,
                                        img_width = args.img_width, img_height = args.img_height)
        train_dataset, test_dataset = dataloader.load_dataset()
        
        model = Pix2Pix()
        model.build_model(arch_gen = args.arch_gen, arch_disc = args.arch_disc,
                          output_channels = args.output_channels)
        model.configure_losses()
        model.configure_optimizers()
        model.configure_checkpoints(checkpoint_path = args.checkpoint_path)
        
        generate_outputs_gan(0, model.generator, test_dataset, output_path = args.output_path)
        for ckpt in model.get_checkpoints():
            model.restore_checkpoint(ckpt)
            epoch = int(model.checkpoint.step)
            generate_outputs_gan(epoch, model.generator, test_dataset, output_path = args.output_path)
    
    elif args.model == 'cyclegan':
        dataloader = Pix2Pix_DataLoader(data_path = args.data_path, batch_size = args.batch_size,
                                        img_width = args.img_width, img_height = args.img_height)
        train_dataset, test_dataset = dataloader.load_dataset()
        
        model = CycleGAN()
        model.build_model(arch_gen = args.arch_gen, arch_disc = args.arch_disc,
                          output_channels = args.output_channels)
        model.configure_losses()
        model.configure_optimizers()
        model.configure_checkpoints(checkpoint_path = args.checkpoint_path)
        
        generate_outputs_gan(0, model.generator_g, test_dataset, output_path = args.output_path)
        for ckpt in model.get_checkpoints():
            model.restore_checkpoint(ckpt)
            epoch = int(model.checkpoint.step)
            generate_outputs_gan(epoch, model.generator_g, test_dataset, output_path = args.output_path)
                  
    elif args.model in ['neural_style_transfer', 'fast_neural_style_transfer']:
        content_paths = glob.glob(os.path.join(args.data_path, 'valA/*.jpg'))
        style_paths = glob.glob(os.path.join(args.data_path, 'trainB/*.jpg'))
        generate_outputs_neural_style_transfer(content_paths, style_paths,
                                               output_path = args.output_path, args = args)
        

if __name__ == '__main__':
    main()
