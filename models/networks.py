'''
This program implements a Pix2Pix network.

References:
    https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    https://www.tensorflow.org/tutorials/generative/pix2pix
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import os, sys
import numpy as np


class InstanceNormalization(keras.layers.Layer):
    ''' An Instance Normalization Layer. '''

    def __init__(self, epsilon=1e-5):
        ''' Initializes the class. '''
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        ''' Builds the layer parameters. '''
        self.scale = self.add_weight(
                      name='scale',
                      shape=input_shape[-1:],
                      initializer=tf.random_normal_initializer(1., 0.02),
                      trainable=True)

        self.offset = self.add_weight(
                      name='offset',
                      shape=input_shape[-1:],
                      initializer='zeros',
                      trainable=True)

    def call(self, x):
        ''' Performs a forward pass. '''
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class Pix2Pix_Net(object):
    ''' A Pix2Pix network. '''

    def __init__(self):
        ''' Initializes the class. '''
        pass
        
    def Generator(self, output_channels, arch = 'unet', norm_type='batchnorm'):
        ''' Generator model. '''
        if arch == 'unet':
            return self.UNet_Generator(output_channels=output_channels, norm_type=norm_type)
        raise Exception("Invalid architecture.")
    
    def Discriminator(self, arch = 'patchgan', norm_type='batchnorm', target=True):
        ''' Discriminator model. '''
        if arch == 'patchgan':
            return self.PatchGAN_Discriminator(norm_type=norm_type, target=target)
        raise Exception("Invalid architecture.")
        
    def UNet_Generator(self, output_channels, norm_type='batchnorm'):
        ''' Modified UNet generator model. '''

        down_stack = [
            self.downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
            self.downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
            self.downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
            self.downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
            self.downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
            self.downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
            self.downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
            self.upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
            self.upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
            self.upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
            self.upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = keras.layers.Conv2DTranspose(
            output_channels, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh')  # (bs, 256, 256, 3)

        inputs = keras.layers.Input(shape=[None, None, 3])
        x = inputs

        # Downsamples through the model.
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsamples and establishes the skip connections.
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = keras.layers.Concatenate()([x, skip])

        x = last(x)

        return keras.Model(inputs=inputs, outputs=x)
        
    
    def PatchGAN_Discriminator(self, norm_type='batchnorm', target=True):
        ''' PatchGAN discriminator model. '''

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = keras.layers.Input(shape=[None, None, 3], name='input_image')
        x = inp

        if target:
            tar = keras.layers.Input(shape=[None, None, 3], name='target_image')
            x = keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels * 2)

        down1 = self.downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = keras.layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer,
            use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        if norm_type.lower() == 'batchnorm':
            norm1 = keras.layers.BatchNormalization()(conv)
        elif norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)

        leaky_relu = keras.layers.LeakyReLU()(norm1)

        zero_pad2 = keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = keras.layers.Conv2D(
            1, 4, strides=1,
            kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        if target:
            return keras.Model(inputs=[inp, tar], outputs=last)
        else:
            return keras.Model(inputs=inp, outputs=last)
    
    def downsample(self, filters, size, norm_type='batchnorm', apply_norm=True):
        ''' Downsample Sequential Model. '''
        
        initializer = tf.random_normal_initializer(0., 0.02)

        result = keras.Sequential()
        result.add(
            keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                result.add(keras.layers.BatchNormalization())
            elif norm_type.lower() == 'instancenorm':
                result.add(InstanceNormalization())

        result.add(keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, norm_type='batchnorm', apply_dropout=False):
        ''' Upsample Sequential Model. '''

        initializer = tf.random_normal_initializer(0., 0.02)

        result = keras.Sequential()
        result.add(
            keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(keras.layers.Dropout(0.5))

        result.add(keras.layers.ReLU())

        return result







