from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

ngf = 64
EPS = 1e-12
l1_weight = 100
gan_weight = 1
lr = 0.0002
beta = 0.5
separable_conv = "store_true"


class Pix2PixVae:
    def __init__(self, z_size=512, batch_size=100, learning_rate=0.0001, is_training=True, reuse=False, gpu_mode=True):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.reuse = reuse
        self.gpu_mode = gpu_mode

        self.g = tf.Graph()
        with self.g.as_default():
            self.input = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
            self.output = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])

        with tf.variable_scope("pix2pix_vae", reuse=self.reuse):
            with self.g.as_default():
                if not self.gpu_mode:
                    with tf.device('/cpu:0'):
                        self.make_model()
                else:
                    self.make_model()
                self.init_session()

    def init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def batchnorm(self, inputs):
        return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

    def lrelu(self, x, a):
        with tf.name_scope('lrelu'):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: a*x/2 - a*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def gen_conv(self, batch_input, out_channels):

        #  MAYBE DO SEPARABLE CONVS??

        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)

    def gen_deconv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if separable_conv:
            _b, h, w, _c = batch_input.shape
            resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1),
                                              padding="same", depthwise_initializer=initializer,
                                              pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                              kernel_initializer=initializer)

    def discrim_conv(self, batch_input, out_channels, stride):
        padded_ip = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv2d(padded_ip, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                                kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def create_generator(self, generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = self.gen_conv(generator_inputs, ngf)
            layers.append(output)

        layer_specs = [ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                       ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                       ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                       ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                       ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                       ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                       ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
                       ]
        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = self.lrelu(layers[-1], 0.2)
                convolved = self.gen_conv(rectified, out_channels)
                output = self.batchnorm(convolved)
                layers.append(output)
        self.encoding = layers[-1]

        layer_specs = [
            (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        print(layers, len(layers))
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1

            print(num_encoder_layers, decoder_layer, skip_layer)

            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    input_ = layers[-1]
                else:
                    input_ = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input_)
                output = self.gen_deconv(rectified, out_channels)
                output = self.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        with tf.variable_scope("decoder_1"):
            input_ = tf.concat([layers[-1], layers[0]], axis=3)
            print(layers[-1], layers[0])
            rectified = tf.nn.relu(input_)
            output = self.gen_deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        print("LAYERS LENGTH ", len(layers), layers[0], layers[-1])
        return layers[-1]

    def create_discriminator(self, discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        input_ = tf.concat([discrim_inputs, discrim_targets], axis=3)

        with tf.variable_scope("layer_1"):
            convolved = self.discrim_conv(input_, ngf, stride=2)
            rectified = self.lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ngf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers else 2
                convolved = self.discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = self.batchnorm(convolved)
                rectified = self.lrelu(normalized, 0.2)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = self.discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    def make_model(self):
        inputs = self.input
        targets = self.output
        with tf.variable_scope('generator'):
            out_channels = int(targets.get_shape()[-1])

            ouputs = self.create_generator(inputs, out_channels)
            print("OUT_CHANNELS ", ouputs)

        with tf.name_scope("real_discrim"):
            with tf.variable_scope("discriminator"):
                predict_real = self.create_discriminator(inputs, targets)

        with tf.name_scope('fake_discrim'):
            with tf.variable_scope("discriminator", reuse=True):
                predict_fake = self.create_discriminator(inputs, ouputs)

        with tf.name_scope("discriminator_loss"):
            self.discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        with tf.name_scope("generator_loss"):
            gen_loss_gan = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_l1 = tf.reduce_mean(tf.abs(targets - ouputs))
            self.gen_loss = gen_loss_gan * gan_weight + gen_loss_l1 * l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_vars = [var for var in tf.trainable_variables() if "discrim" in var.name]
            discrim_opt = tf.train.AdadeltaOptimizer(lr, beta)
            discrim_grads_and_vars = discrim_opt.compute_gradients(self.discrim_loss, var_list=discrim_vars)
            self.discrim_train = discrim_opt.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope('generator_train'):
            with tf.control_dependencies([self.discrim_train]):
                gen_vars = [var for var in tf.trainable_variables() if "generator" in var.name]
                gen_opt = tf.train.AdadeltaOptimizer(lr, beta)
                gen_grads_and_var = gen_opt.compute_gradients(loss=self.gen_loss, var_list=gen_vars)
                self.gen_train = gen_opt.apply_gradients(gen_grads_and_var)

        self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_losses = self.ema.apply([self.discrim_loss, self.gen_loss_GAN, self.gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        self.incr_global_step = tf.assign(global_step, global_step + 1)



