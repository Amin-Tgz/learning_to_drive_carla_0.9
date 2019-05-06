import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images, get_image
from utils import CelebA
import numpy as np
import cv2
from tensorflow.python.framework.ops import convert_to_tensor
import os
import json

TINY = 1e-8
d_scale_factor = 0.25
g_scale_factor = 1 - 0.75 / 2


class VaeGan:
    def __init__(self, z_size=512, batch_size=100, learning_rate=1e-4, kl_tolerance=0.5, is_training=True, reuse=False,
                 gpu_mode=True, log_dir='logs/'):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.kl_tolerance = kl_tolerance
        self.reuse = reuse
        self.variables_to_log = []
        self.log_dir = log_dir

        with tf.variable_scope('vae_gan', reuse=self.reuse):
            if not gpu_mode:
                tf.device('/cpu:0')
                self._build_graph()
            else:
                self._build_graph()
        self._init_session()

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        print("Session initialized.")

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.ep = tf.random_normal(shape=[self.batch_size, self.z_size])
            self.zp = tf.random_normal(shape=[self.batch_size, self.z_size])

            self.input_batch = tf.placeholder(tf.float32, shape=[None, 80, 160, 3])
            self.input_for_generation = tf.placeholder(tf.float32, shape=[1, self.z_size])
            self.mu, self.log_var = self._encode(inp=self.input_batch, reuse=False)
            self.z_input = tf.add(self.mu, tf.sqrt(tf.exp(self.log_var)) * self.ep)

            if not self.is_training:
                self.generated_image_from_code = self._decode(self.input_for_generation, reuse=False)

            if self.is_training:
                self.generated_image = self._decode(z=self.z_input, reuse=False)
                self.generated_conv_discrim, self.generated_discrim_op = self._discriminate(inp=self.generated_image)

                self.fake_image = self._decode(z=self.zp, reuse=True)

                self.real_conv_discrim, self.real_discrim_op = self._discriminate(inp=self.input_batch, reuse=True)
                _, self.fake_discrim_op = self._discriminate(inp=self.fake_image, reuse=True)

                # LOSSES
                # KL Loss
                self.kl_loss = self.KL_loss(self.mu, self.log_var)

                # Discriminator Losses
                self.fake_D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(self.fake_discrim_op), logits=self.fake_discrim_op))
                self.real_D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.real_discrim_op) - d_scale_factor,
                    logits=self.real_discrim_op))  # SEE IN PAPER THE SCALE FACTOR!!!
                self.generated_D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(self.generated_discrim_op), logits=self.generated_discrim_op))

                # Genertor Losses
                self.fake_G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.fake_discrim_op) - g_scale_factor, logits=self.fake_discrim_op))
                self.generated_G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.generated_discrim_op) - g_scale_factor, logits=self.generated_discrim_op))

                # Final Discriminator Loss
                self.D_loss = self.fake_D_loss + self.real_D_loss + self.generated_D_loss

                # Feature Loss
                self.LL_loss = tf.reduce_mean(
                    tf.reduce_sum(self.NLLNormal(self.generated_conv_discrim, self.real_conv_discrim), [1, 2, 3]))

                # Encoder Loss
                self.encode_loss = self.kl_loss / (self.z_size * self.batch_size) - self.LL_loss / (
                    3 * 8 * 256)  # WHY THIS???

                # Generator Loss
                self.G_loss = self.fake_G_loss + self.generated_G_loss - 1e-6 * self.LL_loss

                self.variables_to_log.append(("Encoer_loss", self.encode_loss))
                self.variables_to_log.append(("Gen_loss", self.G_loss))
                self.variables_to_log.append(("Discriminator_loss", self.D_loss))
                self.variables_to_log.append(("LL_loss", self.LL_loss))

                t_vars = tf.trainable_variables()
                self.discrim_vars = [v for v in t_vars if 'Dis' in v.name]
                self.generator_vars = [v for v in t_vars if "Dec" in v.name]
                self.encoder_vars = [v for v in t_vars if "Enc" in v.name]

                self.saver = tf.train.Saver()
                for k, v in self.variables_to_log:
                    tf.summary.scalar(k, v)

                # Declare trainig ops here
                self.global_step = tf.Variable(0, trainable=False)
                self.add_global = self.global_step.assign_add(1)
                self.new_lr = tf.train.exponential_decay(self.learning_rate, global_step=self.global_step,
                                                         decay_steps=10000,
                                                         decay_rate=0.98)

                # For Discrim
                trainer_D = tf.train.RMSPropOptimizer(learning_rate=self.new_lr)
                gradients_D = trainer_D.compute_gradients(self.D_loss, var_list=self.discrim_vars)
                self.optimize_D = trainer_D.apply_gradients(gradients_D)

                # For Generator
                trainer_G = tf.train.RMSPropOptimizer(learning_rate=self.new_lr)
                gradients_G = trainer_G.compute_gradients(self.G_loss, var_list=self.generator_vars)
                self.optimize_G = trainer_G.apply_gradients(gradients_G)

                # Trainer Encoder
                trainer_E = tf.train.RMSPropOptimizer(learning_rate=self.new_lr)
                gradients_E = trainer_E.compute_gradients(self.encode_loss, var_list=self.encoder_vars)
                self.optimize_E = trainer_E.apply_gradients(gradients_E)

            self.init = tf.global_variables_initializer()

    def train(self, feed_dict):
        self.sess.run(self.optimize_E, feed_dict=feed_dict)
        self.sess.run(self.optimize_G, feed_dict=feed_dict)
        self.sess.run(self.optimize_D, feed_dict=feed_dict)

        # summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
        # self.summary_writer.add_summary(summary_str, self.global_step)
        self.sess.run(self.add_global)

        if np.mod(self.sess.run(self.global_step), 50) == 1:
            (E_loss, D_loss, G_loss, LL_loss, KL_loss, new_lr, step) = self.sess.run(
                [self.encode_loss, self.D_loss, self.G_loss, self.LL_loss, self.kl_loss, self.new_lr, self.global_step],
                feed_dict=feed_dict)
            print("Step %d: D: loss = %.7f G: loss=%.7f E: loss=%.7f LL loss=%.7f KL=%.7f, LR=%.7f" % (
                step, D_loss, G_loss, E_loss, LL_loss, KL_loss, new_lr))

    def _encode(self, inp, reuse=False):
        with tf.variable_scope("Encoder") as scope:
            if reuse:
                scope.reuse_variables()
            h = tf.layers.conv2d(inp, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
            h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
            h = tf.reshape(h, [-1, 3 * 8 * 256])

            self.mu = tf.layers.dense(h, self.z_size, name="enc_mu")
            self.log_var = tf.layers.dense(h, self.z_size, name="enc_log_var")
            self.sigma = tf.exp(self.log_var / 2.0)
            self.eps = tf.random.normal([self.batch_size, self.z_size])

            return self.mu, self.log_var

    def _decode(self, z, reuse=False):
        with tf.variable_scope("Decoder") as scope:
            if reuse:
                scope.reuse_variables()
            h = tf.layers.dense(z, 3 * 8 * 256, name="dec_fc")
            h = tf.reshape(h, [-1, 3, 8, 256])
            h = tf.layers.conv2d_transpose(h, 128, 4, strides=2, activation=tf.nn.relu, name="dec_deconv1")
            h = tf.layers.conv2d_transpose(h, 64, 4, strides=2, activation=tf.nn.relu, name="dec_deconv2")
            h = tf.layers.conv2d_transpose(h, 32, 5, strides=2, activation=tf.nn.relu, name="dec_deconv3")
            y = tf.layers.conv2d_transpose(h, 3, 4, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")
            return y

    def _discriminate(self, inp, reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h = tf.layers.conv2d(inp, 32, 4, strides=2, activation=tf.nn.relu, name="dis_conv1")
            h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="dis_conv2")
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="dis_conv3")
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="dis_conv4")
            self.middle_conv = h  # Conv 4 output
            h = tf.reshape(h, [-1, 3 * 8 * 256])

            fl = tf.layers.dense(inputs=h, units=256, name="fc_discriminator")
            self.discrim_op = tf.layers.dense(inputs=fl, units=1, activation=tf.nn.sigmoid)

            return self.middle_conv, self.discrim_op

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def NLLNormal(self, pred, target):
        print("NLL_NORMAL ", pred, target)
        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c

        return tmp

    def _parse_function(self, images_filenames):

        image_string = tf.read_file(images_filenames)
        image_decoded = tf.image.decode_and_crop_jpeg(image_string, crop_window=[218 / 2 - 54, 178 / 2 - 54, 108, 108],
                                                      channels=3)
        image_resized = tf.image.resize_images(image_decoded, [self.output_size, self.output_size])
        image_resized = image_resized / 127.5 - 1

        return image_resized

    def _read_by_function(self, filename):

        array = get_image(filename, 108, is_crop=True, resize_w=self.output_size,
                          is_grayscale=False)
        real_images = np.array(array)
        return real_images

    def encode(self, x):
        return self.sess.run(self.z_input, feed_dict={self.input_batch: x})

    def decode(self, z):
        return self.sess.run(self.generated_image_from_code, feed_dict={self.input_for_generation: z})

    def get_encoder_params(self):
        model_names = []
        model_params = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                if "Enc" in var.name:
                    param_name = var.name
                    p = self.sess.run(param_name)
                    model_names.append(param_name)
                    params = np.round(p * 10000).astype(np.int).tolist()
                    model_params.append(params)
        return model_params, model_names

    def set_encoder_params(self, params, param_names):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                if "Enc" in var.name:
                    pshape = self.sess.run(var).shape
                    p = np.array(params[param_names.index(var.name)])
                    assert pshape == p.shape
                    assign_op = var.assign(p.astype(np.float) / 10000.)
                    self.sess.run(assign_op)

    def save_encoder_json(self, jsonfile="vae_gan.json"):
        encoder_params, encoder_param_names = self.get_encoder_params()
        qparams = []
        qpram_names = []
        for p in encoder_params:
            qparams.append(p)
        for n in encoder_param_names:
            qpram_names.append(n)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ':'))
        with open("vae_gan_param_names.json", 'wt') as outfile:
            json.dump(qpram_names, outfile, sort_keys=True, indent=0, separators=(',', ':'))

    def load_encoder_json(self, jsonfile='vae_gan.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        with open("vae_gan_param_names.json", 'r') as f:
            param_names = json.load(f)
        self.set_encoder_params(params, param_names)
