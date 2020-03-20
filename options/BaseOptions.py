'''
This program implements a template class for command line options.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import argparse


model_options = ['neural_style_transfer', 'fast_neural_style_transfer',
                 'pix2pix', 'cyclegan']
norm_options = [None, 'batchnorm', 'instancenorm']
arch_gen_options = ['unet']
arch_disc_options = ['patchgan']


class BaseOptions(object):
    ''' A template class for command line options. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        self.parser = argparse.ArgumentParser(description = 'GANime: Generating Anime Characters from Sketches.',
                                              formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.add_options(self.parser)
        
    def parse(self):
        ''' Parses command line arguments. '''
        args = self.parser.parse_known_args()[0]
        return args

    def add_options(self, parser):
        ''' Adds command line options. '''
        parser.add_argument('--model', type = str, default = 'pix2pix',
                            help = 'Model.', choices = model_options)
        parser.add_argument('--arch-gen', type = str, default = 'unet',
                            help = 'Architecture for generators.', choices = arch_gen_options)
        parser.add_argument('--arch-disc', type = str, default = 'patchgan',
                            help = 'Architecture for discriminators.', choices = arch_disc_options)
        parser.add_argument('--norm', type=str, default = 'batchnorm',
                            help = 'Type of normalization.')
        parser.add_argument('--data-path', type = str, default = None,
                            help = 'Path to dataset.')
        parser.add_argument('--checkpoint-path', type = str, default = 'checkpoints',
                            help = 'Path to checkpoints.')
        parser.add_argument('--log-path', type = str, default = 'logs',
                            help = 'Path to logs.')
        parser.add_argument('--output-path', type = str, default = 'outputs',
                            help = 'Path to output folder.')
        parser.add_argument('--batch-size', type = int, default = 32,
                            help = 'Size of a minibatch.')
        parser.add_argument('--seed', type = int, default = 0,
                            help = 'Random seed for TensorFlow.')
        
        return parser
    
