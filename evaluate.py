'''
This program evaluates a generative model for line art colorization.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import os, sys
import glob
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('.')
sys.path.append('./data')
sys.path.append('./dataloaders')
sys.path.append('./models')
sys.path.append('./options')
sys.path.append('./utils')


from options.EvaluateOptions import EvaluateOptions
from utils.evaluation_metrics import FID, SSIM


IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


def parseArgs():
    ''' Reads command line arguments. '''
    args = EvaluateOptions().parse()
    args.output_channels = OUTPUT_CHANNELS
    args.img_width = IMG_WIDTH
    args.img_height = IMG_HEIGHT
    return args
    

def save_scores(score_list, score_path):
    ''' Saves the scores. '''
    with open(score_path, 'w') as fp:
        for item in score_list:
            fp.write("%s\n" % item)
            
            
def load_scores(score_path):
    ''' Loads the scores. '''
    with open(score_path, 'r') as fp:
        score_list = [float(line.strip()) for line in fp]
    return score_list
    
    
def visualize_scores(score_list, metric, model, epoch_range):
    ''' Visualizes the scores. '''
    model_names = {
        'neural_style_transfer': 'Neural Style Transfer',
        'fast_neural_style_transfer': 'Fast Neural Style Transfer',
        'pix2pix': 'Pix2Pix',
        'cyclegan': 'CycleGAN',
    }
    
    fig = plt.figure()
    ax = plt.axes()
    if metric == 'fid':
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    elif metric == 'ssim':
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        
    plt.plot(epoch_range, score_list)
    plt.xlabel('Numbers of Epochs')
    plt.ylabel(metric.upper() + ' Score')
    plt.title(metric.upper() + ' Scores for ' + model_names[model] + ' Trained for Various Epochs')
    fig.savefig(metric + '_' + model + '.jpg')
    
    
def main():
    ''' Main program. '''
    args = parseArgs()
        
    if args.metric == 'fid':
        metric = FID()
    elif args.metric == 'ssim':
        metric = SSIM()
        
    if args.model in ['neural_style_transfer', 'fast_neural_style_transfer']:
        data_path_real = os.path.join(args.output_path, 'real')
        data_path_fake = os.path.join(args.output_path, 'fake')
        score = metric.evaluate(data_path_real, data_path_fake)
        score_list = [score]
        
    elif args.model in ['pix2pix', 'cyclegan']:
        score_list = []
        for epoch in range(args.start_epoch, args.epochs + 1, args.save_ckpt_freq):
            epoch_path = os.path.join(args.output_path, 'Epoch {}'.format(epoch))
            if os.path.isdir(epoch_path):
                data_path_real = os.path.join(epoch_path, 'real')
                data_path_fake = os.path.join(epoch_path, 'fake')
                score = metric.evaluate(data_path_real, data_path_fake)
                score_list.append(score)
            else:
                score_list.append(-1)
    
    save_scores(score_list, score_path = args.metric + '_' + args.model + '.txt')
    visualize_scores(score_list, metric = args.metric, model = args.model,
                     epoch_range = range(args.start_epoch, args.epochs + 1, args.save_ckpt_freq))


if __name__ == '__main__':
    main()
