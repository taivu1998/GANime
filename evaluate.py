'''
This program evaluates a generative model for line art colorization.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
matplotlib.use('Agg')

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
from utils.runtime import configure_seeds, ensure_parent_dir, validate_existing_dir


IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


def parseArgs():
    ''' Reads command line arguments. '''
    args = EvaluateOptions().parse()
    args.output_channels = OUTPUT_CHANNELS
    args.img_width = IMG_WIDTH
    args.img_height = IMG_HEIGHT
    validate_existing_dir(args.output_path, '--output-path')
    return args
    

def save_scores(score_list, score_path):
    ''' Saves the scores. '''
    ensure_parent_dir(score_path)
    with open(score_path, 'w') as fp:
        for item in score_list:
            fp.write("%s\n" % item)
            
            
def load_scores(score_path):
    ''' Loads the scores. '''
    with open(score_path, 'r') as fp:
        score_list = [float(line.strip()) for line in fp]
    return score_list
    
    
def visualize_scores(score_list, metric, model, epoch_range, plot_path):
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
    ensure_parent_dir(plot_path)
    fig.savefig(plot_path)
    
    
def main():
    ''' Main program. '''
    args = parseArgs()
    configure_seeds(args.seed)
        
    if args.metric == 'fid':
        metric = FID()
    elif args.metric == 'ssim':
        metric = SSIM()

    report_path = args.report_path or os.path.join(args.output_path, args.metric + '_' + args.model + '.txt')
    plot_path = args.plot_path or os.path.join(args.output_path, args.metric + '_' + args.model + '.jpg')
        
    if args.model in ['neural_style_transfer', 'fast_neural_style_transfer']:
        data_path_real = os.path.join(args.output_path, 'real')
        data_path_fake = os.path.join(args.output_path, 'fake')
        score = metric.evaluate(data_path_real, data_path_fake)
        score_list = [score]
        epoch_range = [0]
        
    elif args.model in ['pix2pix', 'cyclegan']:
        score_list = []
        epoch_range = list(range(args.start_epoch, args.epochs + 1, args.save_ckpt_freq))
        for epoch in epoch_range:
            epoch_path = os.path.join(args.output_path, 'Epoch {}'.format(epoch))
            if os.path.isdir(epoch_path):
                data_path_real = os.path.join(epoch_path, 'real')
                data_path_fake = os.path.join(epoch_path, 'fake')
                score = metric.evaluate(data_path_real, data_path_fake)
                score_list.append(score)
            else:
                score_list.append(-1)
    
    save_scores(score_list, score_path = report_path)
    visualize_scores(score_list, metric = args.metric, model = args.model,
                     epoch_range = epoch_range, plot_path = plot_path)


if __name__ == '__main__':
    main()
