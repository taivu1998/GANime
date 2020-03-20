'''
This program trains a generative model for line art colorization.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import os, sys
import time
import datetime
import glob
import numpy as np


sys.path.append('.')
sys.path.append('./data')
sys.path.append('./dataloaders')
sys.path.append('./models')
sys.path.append('./options')
sys.path.append('./utils')


from options.TrainOptions import TrainOptions

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
    args = TrainOptions().parse()
    args.output_channels = OUTPUT_CHANNELS
    args.img_width = IMG_WIDTH
    args.img_height = IMG_HEIGHT
    return args
    
    
def main():
    ''' Main program. '''
    args = parseArgs()
    
    if args.model == 'pix2pix':
        dataloader = Pix2Pix_DataLoader(data_path = args.data_path, buffer_size = args.buffer_size,
                                        batch_size = args.batch_size, img_width = args.img_width,
                                        img_height = args.img_height)
        train_dataset, test_dataset = dataloader.load_dataset()
        model = Pix2Pix()
        model.build_model(arch_gen = args.arch_gen, arch_disc = args.arch_disc,
                          output_channels = args.output_channels)
        # model.plot_model()    # Uncomment to visualize the model.
        model.configure_losses(lambda_l1_loss = args.lambda_l1_loss, lambda_tv_loss = args.lambda_tv_loss,
                               use_tv_loss = args.use_tv_loss, norm_tv_loss = args.norm_tv_loss)
        model.configure_optimizers(lr = args.lr, optim = args.optim, momentum = args.momentum,
                                   beta_1 = args.beta_1, beta_2 = args.beta_2,
                                   epsilon = args.epsilon, rho = args.rho)
        model.configure_checkpoints(checkpoint_path = args.checkpoint_path)
        model.configure_logs(log_path = args.log_path)
        model.fit(train_dataset, test_dataset, output_path = args.output_path, start_epoch = args.start_epoch,
                  epochs = args.epochs, resume = args.resume, save_ckpt_freq = args.save_ckpt_freq)
    
    elif args.model == 'cyclegan':
        dataloader = CycleGAN_DataLoader(data_path = args.data_path, buffer_size = args.buffer_size,
                                         batch_size = args.batch_size, img_width = args.img_width,
                                         img_height = args.img_height)
        train_sketch, train_color, test_sketch, test_color = dataloader.load_dataset()
        model = CycleGAN()
        model.build_model(arch_gen = args.arch_gen, arch_disc = args.arch_disc,
                          output_channels = args.output_channels)
        # model.plot_model()    # Uncomment to visualize the model.
        model.configure_losses(lambda_cycle_loss = args.lambda_cycle_loss)
        model.configure_optimizers(lr = args.lr, optim = args.optim, momentum = args.momentum,
                                   beta_1 = args.beta_1, beta_2 = args.beta_2,
                                   epsilon = args.epsilon, rho = args.rho)
        model.configure_checkpoints(checkpoint_path = args.checkpoint_path)
        model.configure_logs(log_path = args.log_path)
        model.fit(train_sketch, train_color, test_sketch, test_color, output_path = args.output_path,
                  start_epoch = args.start_epoch, epochs = args.epochs, resume = args.resume,
                  save_ckpt_freq = args.save_ckpt_freq)
                  
    elif args.model == 'neural_style_transfer':
        dataloader = NeuralStyleTransfer_DataLoader(content_path = args.content_path,
                                                    style_path = args.style_path)
        content_image, style_image = dataloader.load_dataset()
        model = NeuralStyleTransfer()
        model.build_model(style_weight = args.style_weight, content_weight = args.content_weight)
        model.configure_optimizers(lr = args.lr, optim = args.optim, momentum = args.momentum,
                                   beta_1 = args.beta_1, beta_2 = args.beta_2,
                                   epsilon = args.epsilon, rho = args.rho)
        model.fit(content_image, style_image, epochs = args.epochs,
                  steps_per_epoch = args.steps_per_epoch, output_path = args.output_path)
    
    elif args.model == 'fast_neural_style_transfer':
        dataloader = NeuralStyleTransfer_DataLoader(content_path = args.content_path,
                                                    style_path = args.style_path)
        content_image, style_image = dataloader.load_dataset()
        model = FastNeuralStyleTransfer()
        model.build_model()
        model.fit(content_image, style_image, output_path = args.output_path)
    
    
if __name__ == '__main__':
    main()
