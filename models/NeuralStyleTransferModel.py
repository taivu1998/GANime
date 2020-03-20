'''
This program implements a Neural Style Transfer model.

References:
    https://www.tensorflow.org/tutorials/generative/style_transfer
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

import os, sys
import time
import numpy as np


from BaseModel import BaseModel
from utils.data_pipeline import *


class NeuralStyleTransfer(BaseModel):
    ''' A Neural Style Transfer model. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        super().__init__()
        
    def build_model(self, style_weight = 1e-2, content_weight = 1e4):
        ''' Builds network architectures. '''
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)
        self.style_weight = style_weight
        self.content_weight = content_weight

    def configure_optimizers(self, lr = 0.02, optim = 'adam', momentum = 0.0,
                             beta_1 = 0.99, beta_2 = 0.999, epsilon = 1e-1, rho = 0.9):
        ''' Configures optimizers. '''
        if optim == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1,
                                                   beta_2 = beta_2, epsilon = epsilon)
  
        elif optim == 'sgd':
            self.optimizer = keras.optimizers.SGD(learning_rate = lr, momentum = momentum)
            
        elif optim == 'rmsprop':
            self.optimizer = keras.optimizers.RMSprop(learning_rate = lr, rho = rho,
                                                      momentum = momentum, epsilon = epsilon)
                             
    def fit(self, content_image, style_image, epochs = 1000,
            steps_per_epoch = 100, output_path = 'stylized_image.png'):
        ''' Trains the model. '''
        self.style_targets = self.extractor(style_image)['style']
        self.content_targets = self.extractor(content_image)['content']
        self.stylized_image = tf.Variable(content_image)
        
        start = time.time()
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(self.stylized_image)
                print('.', end = '')

            self.save_output(output_path)
            print("Train step: {}".format(step))
          
        end = time.time()
        print("Total time: {:.1f}".format(end - start))
    
    @tf.function()
    def train_step(self, image):
        ''' Executes a training step. '''
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))
        
    def style_content_loss(self, outputs):
        ''' Calculates style content loss. '''
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss
        
    def predict(self):
        ''' Generates an output image from an input. '''
        return self.stylized_image
    
    def save_output(self, img_path):
        ''' Saves the output image. '''
        output = tensor_to_image(self.stylized_image)
        output.save(img_path)
    
    def clip_0_1(self, image):
        ''' Clips pixel values. '''
        return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
        

class StyleContentModel(keras.models.Model):
    ''' A Style Content Model. '''

    def __init__(self, style_layers, content_layers):
        ''' Initializes the class. '''
        super(StyleContentModel, self).__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        ''' Performs a forward pass. '''
        # Expects float input in [0, 1].
        inputs = inputs * 255.0
        preprocessed_input = keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name : value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name : value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
        
    def vgg_layers(self, layer_names):
        ''' Creates a VGG model that returns a list of intermediate output values. '''
        vgg = keras.applications.VGG19(include_top = False, weights = 'imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = Model([vgg.input], outputs)
        return model
        
    def gram_matrix(self, input_tensor):
        ''' Calculates gram matrix. '''
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)
