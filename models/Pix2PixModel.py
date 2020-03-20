'''
This program implements a Pix2Pix model.

References:
    https://www.tensorflow.org/tutorials/generative/pix2pix
    https://www.tensorflow.org/tutorials/generative/style_transfer
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

import os, sys
import time
import datetime
import glob
import numpy as np


from BaseModel import BaseModel
from networks import Pix2Pix_Net
from utils.data_pipeline import *


class Pix2Pix(BaseModel):
    ''' A Pix2Pix model. '''

    def __init__(self):
        ''' Initializes the class. '''
        super().__init__()

    def build_model(self, arch_gen = 'unet', arch_disc = 'patchgan', output_channels = 3):
        ''' Builds network architectures. '''
        net = Pix2Pix_Net()
        self.generator = net.Generator(output_channels = output_channels,
                                       arch = arch_gen, norm_type = 'batchnorm')
        self.discriminator = net.Discriminator(arch = arch_disc, norm_type = 'batchnorm', target = True)

    def configure_losses(self, lambda_l1_loss = 100., lambda_tv_loss = 1e-4,
                         use_tv_loss = False, norm_tv_loss = 'l1'):
        ''' Configures losses. '''
        losses = Pix2Pix_Losses()
        self.generator_loss = losses.generator_loss(lambda_l1_loss = lambda_l1_loss,
                                                    lambda_tv_loss = lambda_tv_loss,
                                                    use_tv_loss = use_tv_loss,
                                                    norm_tv_loss = norm_tv_loss)
        self.discriminator_loss = losses.discriminator_loss()

    def configure_optimizers(self, lr = 2e-4, optim = 'adam', momentum = 0.0,
                             beta_1 = 0.5, beta_2 = 0.999, epsilon = 1e-7, rho = 0.9):
        ''' Configures optimizers. '''
        if optim == 'adam':
            self.generator_optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1,
                                                             beta_2 = beta_2, epsilon = epsilon)
            self.discriminator_optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1,
                                                                 beta_2 = beta_2, epsilon = epsilon)
                                                                 
        elif optim == 'sgd':
            self.generator_optimizer = keras.optimizers.SGD(learning_rate = lr, momentum = momentum)
            self.discriminator_optimizer = keras.optimizers.SGD(learning_rate = lr, momentum = momentum)
            
        elif optim == 'rmsprop':
            self.generator_optimizer = keras.optimizers.RMSprop(learning_rate = lr, rho = rho,
                                                                momentum = momentum, epsilon = epsilon)
            self.discriminator_optimizer = keras.optimizers.RMSprop(learning_rate = lr, rho = rho,
                                                                    momentum = momentum, epsilon = epsilon)

    def configure_checkpoints(self, checkpoint_path):
        ''' Configures checkpoints. '''
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
                                              generator_optimizer = self.generator_optimizer,
                                              discriminator_optimizer = self.discriminator_optimizer,
                                              generator = self.generator,
                                              discriminator = self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path,
                                                             max_to_keep = None)

    def configure_logs(self, log_path):
        ''' Configures logs. '''
        self.train_loss_g = keras.metrics.Mean(name = 'train_loss_g', dtype = tf.float32)
        self.train_loss_d = keras.metrics.Mean(name = 'train_loss_d', dtype = tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(log_path, current_time, 'train')
        test_log_dir = os.path.join(log_path, current_time, 'test')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_path, 'fit', current_time))

    def fit(self, train_dataset, test_dataset, output_path,
            start_epoch = 0, epochs = 150, resume = True, save_ckpt_freq = 5):
        ''' Trains the model. '''
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        latest_ckpt = self.checkpoint_manager.latest_checkpoint
        if resume and latest_ckpt:
            self.checkpoint.restore(latest_ckpt)
            start_epoch = int(self.checkpoint.step)
            print("Model Restored from {}".format(latest_ckpt))

        for epoch in range(start_epoch, epochs):
            start = time.time()
            self.save_output(epoch, train_dataset, test_dataset, output_path)
            
            print("Epoch: ", epoch)
            print("Checkpoint Step: ", int(self.checkpoint.step))

            for n, (input_image, target) in train_dataset.enumerate():
                print('.', end = '')
                if (n + 1) % 100 == 0:
                    print()
                self.train_step(input_image, target, epoch)
            print()

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss_g', self.train_loss_g.result(), step = epoch)
                tf.summary.scalar('loss_d', self.train_loss_d.result(), step = epoch)

            if (epoch + 1) % save_ckpt_freq == 0:
                self.checkpoint.step.assign(epoch + 1)
                save_path = self.checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                time.time() - start))
            
        self.checkpoint.step.assign(epoch + 1)
        save_path = self.checkpoint_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))

    @tf.function
    def train_step(self, input_image, target, epoch):
        ''' Executes a training step. '''
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training = True)

            disc_real_output = self.discriminator([input_image, target], training = True)
            disc_generated_output = self.discriminator([input_image, gen_output], training = True)

            gen_total_loss, gen_gan_loss, gen_l1_loss, gen_tv_loss = \
                self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step = epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step = epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step = epoch)
            tf.summary.scalar('gen_tv_loss', gen_tv_loss, step = epoch)
            tf.summary.scalar('disc_loss', disc_loss, step = epoch)
        
        self.train_loss_g(gen_total_loss)
        self.train_loss_d(disc_loss)

    def save_output(self, epoch, train_dataset, test_dataset, output_path):
        ''' Saves output images for each epoch. '''
        epoch_path = os.path.join(output_path, 'Epoch ' + str(epoch).zfill(3))
        if not os.path.isdir(epoch_path):
            os.mkdir(epoch_path)

        dataset_names = {train_dataset: 'train', test_dataset: 'val'}

        for dataset in [train_dataset, test_dataset]:
            count = 0
            for example_input, example_target in dataset.take(2):
                count += 1
                example_prediction = self.generator(example_input, training = True)
                img_list = [example_input[0], example_target[0], example_prediction[0]]
                img_list = [tf.cast((img + 1) * 127.5, tf.uint8) for img in img_list]
                img_type = ['real_sketch', 'real_color', 'fake_color']

                for i in range(3):
                    img_name = 'epoch' + str(epoch).zfill(3) + '_' + dataset_names[dataset] + \
                                '_img' + str(count) + '_' + img_type[i] + '.jpg'
                    img_path = os.path.join(epoch_path, img_name)
                    image_jpg = tf.io.encode_jpeg(img_list[i])
                    tf.io.write_file(img_path, image_jpg)

    def predict(self, img_input):
        ''' Generates an output image from an input. '''
        return self.generator(img_input)

    def load_checkpoints(self):
        ''' Loads the latest checkpoint. '''
        latest_ckpt = self.checkpoint_manager.latest_checkpoint
        if latest_ckpt:
            self.checkpoint.restore(latest_ckpt)
            print("Model Restored from {}".format(latest_ckpt))

    def plot_model(self):
        ''' Visualizes the network architectures. '''
        keras.utils.plot_model(self.generator, to_file = 'generator.jpg',
                               show_shapes = True, dpi = 64)
        keras.utils.plot_model(self.discriminator, to_file = 'discriminator.jpg',
                               show_shapes = True, dpi = 64)
    
    def get_checkpoints(self):
        ''' Gets the list of checkpoints. '''
        return self.checkpoint_manager.checkpoints
        
    def get_latest_checkpoint(self):
        ''' Gets the latest checkpoint. '''
        return self.checkpoint_manager.latest_checkpoint
        
    def restore_checkpoint(self, ckpt):
        ''' Restores a checkpoint. '''
        self.checkpoint.restore(ckpt)
        print("Model Restored from {}".format(ckpt))
        

class Pix2Pix_Losses(object):
    ''' A class for Pix2Pix losses. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        self.loss_object = keras.losses.BinaryCrossentropy(from_logits = True)
    
    def generator_loss(self, lambda_l1_loss = 100., lambda_tv_loss = 1e-4,
                       use_tv_loss = False, norm_tv_loss = 'l1'):
        ''' Generator loss. '''
        return lambda disc_generated_output, gen_output, target: \
            self.gen_loss(disc_generated_output, gen_output, target,
                          lambda_l1_loss = lambda_l1_loss,
                          lambda_tv_loss = lambda_tv_loss,
                          use_tv_loss = use_tv_loss,
                          norm_tv_loss = norm_tv_loss)
        
    def discriminator_loss(self):
        ''' Discriminator loss. '''
        return self.disc_loss
        
    def gen_loss(self, disc_generated_output, gen_output, target,
                 lambda_l1_loss = 100., lambda_tv_loss = 1e-4,
                 use_tv_loss = False, norm_tv_loss = 'l1'):
        ''' Calculates generator loss. '''
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output),
                                    disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        if use_tv_loss:
            tv_loss = self.total_variation_loss(gen_output, norm_tv_loss = norm_tv_loss)
        else:
            tv_loss = 0

        total_gen_loss = gan_loss + (lambda_l1_loss * l1_loss) + (lambda_tv_loss * tv_loss)
        return total_gen_loss, gan_loss, l1_loss, tv_loss
    
    def total_variation_loss(self, gen_output, norm_tv_loss = 'l1'):
        ''' Calculates total variation loss. '''
        x_deltas = gen_output[:, :, 1:, :] - gen_output[:, :, :-1, :]
        y_deltas = gen_output[:, 1:, :, :] - gen_output[:, :-1, :, :]
        
        if norm_tv_loss == 'l1':
            return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))
            # return tf.reduce_mean(tf.image.total_variation(gen_output))
        elif norm_tv_loss == 'l2':
            return tf.reduce_mean(tf.sqrt(2 * (tf.nn.l2_loss(x_deltas) + tf.nn.l2_loss(y_deltas))))
    
    
    def disc_loss(self, disc_real_output, disc_generated_output):
        ''' Calculates discriminator loss. '''
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
