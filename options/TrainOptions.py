'''
This program implements a class for model training command line options.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import argparse


from BaseOptions import BaseOptions


optimizer_options = ['adam', 'sgd', 'rmsprop']
norm_tv_loss_options = ['l1', 'l2']


class TrainOptions(BaseOptions):
    ''' A class for model training command line options. '''

    def __init__(self):
        ''' Initializes the class. '''
        super().__init__()
        self.parser = self.add_options_train(self.parser)

    def add_options_train(self, parser):
        ''' Adds command line options. '''
        parser.add_argument('--lr', type = float, default = 2e-4,
                            help = 'Learning rate.')
        parser.add_argument('--start-epoch', type = int, default = 0,
                            help = 'Starting epoch.')
        parser.add_argument('--epochs', type = int, default = 150,
                            help = 'Number of epochs.')
        parser.add_argument('--buffer-size', type = int, default = 400,
                            help = 'Buffer size for shuffling.')
        parser.add_argument('--augment', action = 'store_true', default = False,
                            help = 'Augment dataset by flipping and cropping.')
        parser.add_argument('--optim', type = str, default = 'adam',
                            help = 'Optimization algorithm.', choices = optimizer_options)
        parser.add_argument('--momentum', type = float, default = 0.0,
                            help = 'Momentum in optimization algorithm.')
        parser.add_argument('--beta-1', type = float, default = 0.5,
                            help = 'Beta 1 in Adam algorithm.')
        parser.add_argument('--beta-2', type = float, default = 0.999,
                            help = 'Beta 2 in Adam algorithm.')
        parser.add_argument('--epsilon', type = float, default = 1e-7,
                            help = 'Epsilon in optimization algorithm.')
        parser.add_argument('--rho', type = float, default = 0.9,
                            help = 'Rho in RMSprop algorithm.')
        parser.add_argument('--lambda-l1-loss', type = float, default = 100.,
                            help = 'Lambda in GAN L1 loss.')
        parser.add_argument('--use-tv-loss', action = 'store_true', default = False,
                            help = 'Use total variation loss.')
        parser.add_argument('--norm-tv-loss', type = str, default = 'l1',
                            help = 'Norm in variation loss.', choices = norm_tv_loss_options)
        parser.add_argument('--lambda-tv-loss', type = float, default = 1e-4,
                            help = 'Lambda in GAN total variation loss.')
        parser.add_argument('--lambda-cycle-loss', type = float, default = 1e-4,
                            help = 'Lambda in GAN cycle consistency loss.')
        parser.add_argument('--resume', action = 'store_true', default = False,
                            help = 'Resume from checkpoints.')
        parser.add_argument('--save-ckpt-freq', type = int, default = 5,
                            help = 'Frequency of saving checkpoints.')
        parser.add_argument('--content-path', type = str, default = None,
                            help = 'Path to content image.')
        parser.add_argument('--style-path', type = str, default = None,
                            help = 'Path to style image.')
        parser.add_argument('--content-weight', type = float, default = 1e4,
                            help = 'Content weight.')
        parser.add_argument('--style-weight', type = float, default = 1e-2,
                            help = 'Style weight.')
        parser.add_argument('--steps-per-epoch', type = int, default = 100,
                            help = 'Number of steps per epoch.')
        
        return parser
