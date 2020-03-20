'''
This program implements a class for model testing command line options.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import argparse


from BaseOptions import BaseOptions


class TestOptions(BaseOptions):
    ''' A class for model testing command line options. '''

    def __init__(self):
        ''' Initializes the class. '''
        super().__init__()
        self.parser = self.add_options_test(self.parser)

    def add_options_test(self, parser):
        ''' Adds command line options. '''
        return parser
