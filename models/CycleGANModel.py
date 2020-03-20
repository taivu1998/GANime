'''
This program implements a CycleGAN model.

References:
    https://www.tensorflow.org/tutorials/generative/cyclegan
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


class CycleGAN(BaseModel):
    ''' A CycleGAN model. '''

    def __init__(self):
        ''' Initializes the class. '''
        super().__init__()

    def build_model(self, arch_gen = 'unet', arch_disc = 'patchgan', output_channels = 3):
        ''' Builds network architectures. '''
        net = Pix2Pix_Net()
        self.generator_g = net.Generator(output_channels = output_channels,
                                         arch = arch_gen, norm_type = 'instancenorm')
        self.generator_f = net.Generator(output_channels = output_channels,
                                         arch = arch_gen, norm_type = 'instancenorm')
        self.discriminator_x = net.Discriminator(arch = arch_disc, norm_type = 'instancenorm',
                                                 target = False)
        self.discriminator_y = net.Discriminator(arch = arch_disc, norm_type = 'instancenorm',
                                                 target = False)

    def configure_losses(self, lambda_cycle_loss = 10):
        ''' Configures losses. '''
        losses = CycleGAN_Losses()
        self.generator_loss = losses.generator_loss()
        self.discriminator_loss = losses.discriminator_loss()
        self.cycle_loss = losses.cycle_loss(lambda_cycle_loss = lambda_cycle_loss)
        self.identity_loss = losses.identity_loss(lambda_cycle_loss = lambda_cycle_loss)

    def configure_optimizers(self, lr = 2e-4, optim = 'adam', momentum = 0.0,
                             beta_1 = 0.5, beta_2 = 0.999, epsilon = 1e-7, rho = 0.9):
        ''' Configures optimizers. '''
        if optim == 'adam':
            self.generator_g_optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1,
                                                               beta_2 = beta_2, epsilon = epsilon)
            self.generator_f_optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1,
                                                               beta_2 = beta_2, epsilon = epsilon)
            self.discriminator_x_optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1,
                                                                   beta_2 = beta_2, epsilon = epsilon)
            self.discriminator_y_optimizer = keras.optimizers.Adam(learning_rate = lr, beta_1 = beta_1,
                                                                   beta_2 = beta_2, epsilon = epsilon)
                                                                 
        elif optim == 'sgd':
            self.generator_g_optimizer = keras.optimizers.SGD(learning_rate = lr, momentum = momentum)
            self.generator_f_optimizer = keras.optimizers.SGD(learning_rate = lr, momentum = momentum)
            self.discriminator_x_optimizer = keras.optimizers.SGD(learning_rate = lr, momentum = momentum)
            self.discriminator_y_optimizer = keras.optimizers.SGD(learning_rate = lr, momentum = momentum)
            
        elif optim == 'rmsprop':
            self.generator_g_optimizer = keras.optimizers.RMSprop(learning_rate = lr, rho = rho,
                                                                  momentum = momentum, epsilon = epsilon)
            self.generator_f_optimizer = keras.optimizers.RMSprop(learning_rate = lr, rho = rho,
                                                                  momentum = momentum, epsilon = epsilon)
            self.discriminator_x_optimizer = keras.optimizers.RMSprop(learning_rate = lr, rho = rho,
                                                                      momentum = momentum, epsilon = epsilon)
            self.discriminator_y_optimizer = keras.optimizers.RMSprop(learning_rate = lr, rho = rho,
                                                                      momentum = momentum, epsilon = epsilon)

    def configure_checkpoints(self, checkpoint_path):
        ''' Configures checkpoints. '''
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
                                              generator_g = self.generator_g,
                                              generator_f = self.generator_f,
                                              discriminator_x = self.discriminator_x,
                                              discriminator_y = self.discriminator_y,
                                              generator_g_optimizer = self.generator_g_optimizer,
                                              generator_f_optimizer = self.generator_f_optimizer,
                                              discriminator_x_optimizer = self.discriminator_x_optimizer,
                                              discriminator_y_optimizer = self.discriminator_y_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path,
                                                             max_to_keep = None)

    def configure_logs(self, log_path):
        ''' Configures logs. '''
        self.train_loss_g_g = keras.metrics.Mean(name = 'train_loss_g_g', dtype = tf.float32)
        self.train_loss_g_f = keras.metrics.Mean(name = 'train_loss_g_f', dtype = tf.float32)
        self.train_loss_d_x = keras.metrics.Mean(name = 'train_loss_d_x', dtype = tf.float32)
        self.train_loss_d_y = keras.metrics.Mean(name = 'train_loss_d_y', dtype = tf.float32)
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(log_path, current_time, 'train')
        test_log_dir = os.path.join(log_path, current_time, 'test')
        
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_path, 'fit', current_time))

    def fit(self, train_sketch, train_color, test_sketch, test_color, output_path,
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
            self.save_output(epoch, train_sketch, test_sketch, output_path)
            
            print("Epoch: ", epoch)
            print("Checkpoint Step: ", int(self.checkpoint.step))
            
            n = 0
            for image_x, image_y in tf.data.Dataset.zip((train_sketch, train_color)):
                self.train_step(image_x, image_y, epoch)
                if n % 10 == 0:
                    print ('.', end = '')
                n += 1
                
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss_g_g', self.train_loss_g_g.result(), step = epoch)
                tf.summary.scalar('loss_g_f', self.train_loss_g_f.result(), step = epoch)
                tf.summary.scalar('loss_d_x', self.train_loss_d_x.result(), step = epoch)
                tf.summary.scalar('loss_d_y', self.train_loss_d_y.result(), step = epoch)

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
    def train_step(self, real_x, real_y, epoch):
        ''' Executes a training step. '''
        with tf.GradientTape(persistent = True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training = True)
            cycled_x = self.generator_f(fake_y, training = True)

            fake_x = self.generator_f(real_y, training = True)
            cycled_y = self.generator_g(fake_x, training = True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training = True)
            same_y = self.generator_g(real_y, training = True)

            disc_real_x = self.discriminator_x(real_x, training = True)
            disc_real_y = self.discriminator_y(real_y, training = True)

            disc_fake_x = self.discriminator_x(fake_x, training = True)
            disc_fake_y = self.discriminator_y(fake_y, training = True)

            # Calculates the loss.
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss.
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
      
        # Calculates the gradients for generator and discriminator.
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Applies the gradients to the optimizer.
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))
                                                           
        with self.summary_writer.as_default():
            tf.summary.scalar('total_gen_g_loss', total_gen_g_loss, step = epoch)
            tf.summary.scalar('total_gen_f_loss', total_gen_f_loss, step = epoch)
            tf.summary.scalar('disc_x_loss', disc_x_loss, step = epoch)
            tf.summary.scalar('disc_y_loss', disc_y_loss, step = epoch)

        self.train_loss_g_g(total_gen_g_loss)
        self.train_loss_g_f(total_gen_f_loss)
        self.train_loss_d_x(disc_x_loss)
        self.train_loss_d_y(disc_y_loss)

    def save_output(self, epoch, train_dataset, test_dataset, output_path):
        ''' Saves output images for each epoch. '''
        epoch_path = os.path.join(output_path, 'Epoch ' + str(epoch).zfill(3))
        if not os.path.isdir(epoch_path):
            os.mkdir(epoch_path)
            
        dataset_names = {train_dataset: 'train', test_dataset: 'val'}

        for dataset in [train_dataset, test_dataset]:
            count = 0
            for example_input in dataset.take(2):
                count += 1
                example_prediction = self.generator_g(example_input, training = True)
                img_list = [example_input[0], example_prediction[0]]
                img_list = [tf.cast((img + 1) * 127.5, tf.uint8) for img in img_list]
                img_type = ['real_sketch', 'fake_color']

                for i in range(2):
                    img_name = 'epoch' + str(epoch).zfill(3) + '_' + dataset_names[dataset] + \
                                '_img' + str(count) + '_' + img_type[i] + '.jpg'
                    img_path = os.path.join(epoch_path, img_name)
                    image_jpg = tf.io.encode_jpeg(img_list[i])
                    tf.io.write_file(img_path, image_jpg)

    def predict(self, img_input):
        ''' Generates an output image from an input. '''
        return self.generator_g(img_input)

    def load_checkpoints(self):
        ''' Loads the latest checkpoint. '''
        latest_ckpt = self.checkpoint_manager.latest_checkpoint
        if latest_ckpt:
            self.checkpoint.restore(latest_ckpt)
            print("Model Restored from {}".format(latest_ckpt))

    def plot_model(self):
        ''' Visualizes the network architectures. '''
        keras.utils.plot_model(self.generator_g, to_file = 'generator_g.jpg',
                               show_shapes = True, dpi = 64)
        keras.utils.plot_model(self.generator_f, to_file = 'generator_f.jpg',
                               show_shapes = True, dpi = 64)
        keras.utils.plot_model(self.discriminator_x, to_file = 'discriminator_x.jpg',
                               show_shapes = True, dpi = 64)
        keras.utils.plot_model(self.discriminator_y, to_file = 'discriminator_y.jpg',
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


class CycleGAN_Losses(object):
    ''' A class for CycleGAN losses. '''

    def __init__(self):
        ''' Initializes the class. '''
        self.loss_object = keras.losses.BinaryCrossentropy(from_logits = True)
        
    def generator_loss(self):
        ''' Generator loss. '''
        return self.gen_loss
        
    def discriminator_loss(self):
        ''' Discriminator loss. '''
        return self.disc_loss
        
    def cycle_loss(self, lambda_cycle_loss = 10):
        ''' Cycle consistency loss. '''
        return lambda real_image, cycled_image: \
            self.calc_cycle_loss(real_image, cycled_image,
                                 lambda_cycle_loss = lambda_cycle_loss)
                   
    def identity_loss(self, lambda_cycle_loss = 10):
        ''' Identity loss. '''
        return lambda real_image, same_image: \
            self.calc_identity_loss(real_image, same_image,
                                    lambda_cycle_loss = lambda_cycle_loss)
        
    def disc_loss(self, disc_real_output, disc_generated_output):
        ''' Calculates discriminator loss. '''
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5
       
    def gen_loss(self, gen_output):
        ''' Calculates generator loss. '''
        return self.loss_object(tf.ones_like(gen_output), gen_output)
        
    def calc_cycle_loss(self, real_image, cycled_image, lambda_cycle_loss = 10):
        ''' Calcualtes cycle consistency loss. '''
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return lambda_cycle_loss * loss
    
    def calc_identity_loss(self, real_image, same_image, lambda_cycle_loss = 10):
        ''' Calculates identity loss. '''
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return lambda_cycle_loss * 0.5 * loss
