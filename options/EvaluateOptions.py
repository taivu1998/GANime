'''
This program implements a class for model evaluation command line options.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import argparse


from BaseOptions import BaseOptions


metric_options = ['fid', 'ssim']


class EvaluateOptions(BaseOptions):
    ''' A class for model evaluation command line options. '''

    def __init__(self):
        ''' Initializes the class. '''
        super().__init__()
        self.parser = self.add_options_evaluate(self.parser)

    def add_options_evaluate(self, parser):
        ''' Adds command line options. '''
        parser.add_argument('--metric', type = str, default = 'fid',
                            help = 'Evaluation metric.', choices = metric_options)
        parser.add_argument('--start-epoch', type = int, default = 0,
                            help = 'Starting epoch.')
        parser.add_argument('--epochs', type = int, default = 150,
                            help = 'Number of epochs.')
        parser.add_argument('--save-ckpt-freq', type = int, default = 5,
                            help = 'Frequency of saving checkpoints.')
        return parser
