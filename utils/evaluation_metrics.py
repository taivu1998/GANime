'''
This program implements some evaluation metrics.

References:
    https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
    https://www.tensorflow.org/api_docs/python/tf/image/ssim
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

import os
import glob
import numpy as np
import scipy
from PIL import Image


class FID(object):
    ''' A class for FID evaluation metric. '''

    def __init__(self):
        ''' Initializes the class. '''
        self.model = InceptionV3(include_top = False, weights = 'imagenet', input_tensor = None,
                                 input_shape = (256, 256, 3), pooling = 'avg', classes = 1000)
        
    def evaluate(self, data_path_real, data_path_fake):
        ''' Calculates the FID score between two data paths. '''
        real_arr, fake_arr = self.load_images(data_path_real, data_path_fake)
        fid = self.fid_score(real_arr, fake_arr)
        return fid
        
    def load_images(self, data_path_real, data_path_fake):
        ''' Loads two sets of image. '''
        path_list_real = glob.glob(os.path.join(data_path_real, '*.jpg'))
        path_list_fake = glob.glob(os.path.join(data_path_fake, '*.jpg'))
        real_list = []
        fake_list = []
        
        for i in range(len(path_list_real)):
            image_real = Image.open(path_list_real[i])
            data_real = np.asarray(image_real)
            real_list.append(data_real)
            image_fake = Image.open(path_list_fake[i])
            data_fake = np.asarray(image_fake)
            fake_list.append(data_fake)
        
        real_arr = np.asarray(real_list)
        fake_arr = np.asarray(fake_list)
        
        return (real_arr, fake_arr)
        
    def fid_score(self, images1, images2):
        ''' Calcualtes the FID score between two sets of images. '''
        images1 = images1.astype('float32')
        images2 = images2.astype('float32')
        
        images1 = preprocess_input(images1)
        images2 = preprocess_input(images2)
        fid = self.calculate_fid(images1, images2)
        
        return fid
        
    def calculate_fid(self, images1, images2):
        ''' Calcualtes the FID score between two sets of images. '''
        act1 = self.model.predict(images1)
        act2 = self.model.predict(images2)
        
        mu1, sigma1 = act1.mean(axis = 0), np.cov(act1, rowvar = False)
        mu2, sigma2 = act2.mean(axis = 0), np.cov(act2, rowvar = False)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


class SSIM(object):
    ''' A class for SSIM evaluation metric. '''

    def __init__(self):
        ''' Initializes the class. '''
        pass
        
    def evaluate(self, data_path_real, data_path_fake):
        ''' Calculates the average SSIM score between two data paths. '''
        path_list_real = glob.glob(os.path.join(data_path_real, '*.jpg'))
        path_list_fake = glob.glob(os.path.join(data_path_fake, '*.jpg'))
        path_list_real = sorted(path_list_real)
        path_list_fake = sorted(path_list_fake)
        
        score_list = []
        for i in range(len(path_list_real)):
            score = self.ssim_score(path_list_real[i], path_list_fake[i])
            score_list.append(score)
            
        return sum(score_list) / len(score_list)
        
    def ssim_score(self, img_path_real, img_path_fake):
        ''' Calculates the SSIM score between two image paths. '''
        img_real = tf.io.read_file(img_path_real)
        img_real = tf.image.decode_png(img_real)
        img_fake = tf.io.read_file(img_path_fake)
        img_fake = tf.image.decode_png(img_fake)
        ssim = self.calculate_ssim(img_real, img_fake, dtype = tf.uint8)
        return float(ssim)
        
    def calculate_ssim(self, img_real, img_fake, dtype = tf.uint8):
        ''' Calculates the SSIM score between two images. '''
        if dtype == tf.uint8:
            ssim = tf.image.ssim(img_real, img_fake, max_val = 255, filter_size = 11,
                                 filter_sigma = 1.5, k1 = 0.01, k2 = 0.03)
                                 
        elif dtype == tf.float32:
            img_real = tf.image.convert_image_dtype(img_real, tf.float32)
            img_fake = tf.image.convert_image_dtype(img_fake, tf.float32)
            ssim = tf.image.ssim(img_real, img_fake, max_val = 1.0, filter_size = 11,
                                 filter_sigma = 1.5, k1 = 0.01, k2 = 0.03)
                                 
        return ssim
