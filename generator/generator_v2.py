from math import ceil

import numpy as np
import tensorflow as tf
from keras import backend as K, regularizers
from keras.backend import flatten
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Reshape, Conv2DTranspose, \
    Lambda, Dropout, GaussianNoise, MaxPooling2D, Normalization, BatchNormalization, AveragePooling2D
from keras.legacy_tf_layers.convolutional import Conv2D
from keras.optimizers import Nadam, Adam


class NoisyAdam(Nadam):

    def __init__(self, strength, sustain, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 name='NoisyAdam', **kwargs):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, name=name, **kwargs)
        self.strength = strength
        self.sustain = sustain
        self.epoch = 0

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        if self.strength > 0:
            stddev = self.strength * (self.sustain ** self.epoch)
            layers = len(grads_and_vars)
            grads_and_vars = [
                (
                    tf.add(gradient,
                           tf.random.truncated_normal(stddev=stddev * 1.01 ** (layers - n), mean=0.,
                                                      shape=gradient.shape)),
                    var) for n, (gradient, var) in enumerate(grads_and_vars)]
        return super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)


class Generator(tf.keras.Model):
    def __init__(self, latent_size: int = 512, input_shape: tuple = (None, 80, 80, 3)):
        super().__init__()

        self._mean = None
        self._log_var = None

        class Encoder(tf.keras.Model):
            def __init__(self):
                super(Encoder, self).__init__()
                self._norm = Normalization()
                self._noise_1 = GaussianNoise(0.2)
                self._noise_2 = GaussianNoise(0.1)
                self._noise_3 = GaussianNoise(0.05)
                self._noise_4 = GaussianNoise(0.01)

                self._bn_1 = BatchNormalization()
                self._bn_2 = BatchNormalization()
                self._bn_3 = BatchNormalization()

                self._conv_1 = Conv2D(64, 3, activation='leaky_relu', padding='same')
                self._conv_2 = Conv2D(64, 3, activation='leaky_relu', padding='same')
                self._pool_1 = MaxPooling2D(2, 2)

                self._conv_3 = Conv2D(64, 3, activation='leaky_relu', padding='same')
                self._conv_4 = Conv2D(64, 3, activation='leaky_relu', padding='same')
                self._pool_2 = MaxPooling2D(2, 2)

                self._conv_5 = Conv2D(128, 3, activation='leaky_relu', padding='same')
                self._conv_6 = Conv2D(128, 3, activation='leaky_relu', padding='same')
                self._pool_3 = MaxPooling2D(2, 2)

                self._conv_7 = Conv2D(256, 3, activation='leaky_relu', padding='same')
                self._conv_8 = Conv2D(256, 3, activation='leaky_relu', padding='same')
                self._pool_4 = MaxPooling2D(2, 2)

                self._conv_9 = Conv2D(512, 3, activation='leaky_relu', padding='same')
                self._conv_10 = Conv2D(512, 3, activation='leaky_relu', padding='valid')
                self._pool_5 = MaxPooling2D(2, 2)

            def call(self, inputs, training=None, mask=None):
                if training:
                    inputs = self._noise_1(inputs)
                inputs = self._norm(inputs)
                inputs = self._conv_1(inputs)
                inputs = self._conv_2(inputs)
                inputs = self._pool_1(inputs)

                inputs = self._conv_3(inputs)
                inputs = self._conv_4(inputs)
                inputs = self._pool_2(inputs)
                if training:
                    inputs = self._noise_2(inputs)
                inputs = self._conv_5(inputs)
                inputs = self._conv_6(inputs)
                inputs = self._pool_3(inputs)
                if training:
                    inputs = self._noise_3(inputs)
                inputs = self._conv_7(inputs)
                inputs = self._conv_8(inputs)
                inputs = self._pool_4(inputs)
                if training:
                    inputs = self._noise_4(inputs)

                inputs = self._conv_9(inputs)
                inputs = self._conv_10(inputs)
                return self._pool_5(inputs)

        self.encoder = Encoder()
        # self.encoder.trainable = False
        self._flatten = Flatten()

        self._latent = Lambda(self._compute_latent, output_shape=(latent_size,))
        self._latent_mean = Dense(latent_size, name='latent_mean')
        self._latent_log_var = Dense(latent_size, name='latent_log_var')
        # self._latent_mean.trainable = False
        # self._latent_log_var.trainable = False

        class Decoder(tf.keras.Model):
            def __init__(self, latent_size):
                super(Decoder, self).__init__()

                self._reshape = Reshape(target_shape=(1, 1, latent_size))
                self._generate_0 = Conv2DTranspose(latent_size, 1, 1, activation='leaky_relu',
                                                   kernel_regularizer=regularizers.l2(0.04))
                self._generate_1 = Conv2DTranspose(512, 2, 2, activation='leaky_relu',
                                                   kernel_regularizer=regularizers.l2(0.04))
                self._generate_2 = Conv2DTranspose(256, 2, 2, use_bias=True, activation='leaky_relu')
                self._generate_3 = Conv2DTranspose(128, 2, 2, use_bias=True, activation='leaky_relu',
                                                   kernel_regularizer=regularizers.l2(0.02))
                self._generate_4 = Conv2DTranspose(96, 2, 1, use_bias=True, activation='leaky_relu', padding='same')
                self._generate_5 = Conv2DTranspose(96, 2, 2, use_bias=True, activation='leaky_relu',
                                                   kernel_regularizer=regularizers.l2(0.02))
                self._generate_6 = Conv2DTranspose(72, 2, 1, use_bias=True, activation='leaky_relu', padding='same',
                                                   kernel_regularizer=regularizers.l2())
                self._generate_7 = Conv2DTranspose(64, 2, 2, use_bias=True, activation='leaky_relu',
                                                   kernel_regularizer=regularizers.l2())
                self._generate_8 = Conv2DTranspose(48, 2, 1, use_bias=True, activation='leaky_relu', padding='same',
                                                   kernel_regularizer=regularizers.l2())
                self._output = Conv2DTranspose(3, 2, 2, use_bias=True, activation='sigmoid')

            def call(self, inputs, training=None, mask=None):

                inputs = self._reshape(inputs)  # 1x1xlatent
                inputs = self._generate_0(inputs)  # 1x1xlatent
                inputs = self._generate_1(inputs)  # 2x2x512
                inputs = self._generate_2(inputs)  # 4x4x256

                inputs = self._generate_3(inputs)  # 8x8x128
                inputs = self._generate_4(inputs)  # 8x8x96

                inputs = self._generate_5(inputs)  # 16x16x72
                inputs = self._generate_6(inputs)  # 16x16x64

                inputs = self._generate_7(inputs)  # 32x32x48
                inputs = self._generate_8(inputs)  # 32x32x48
                return self._output(inputs)  # 64x64x3

        self.decoder = Decoder(latent_size)
        #  self.decoder.trainable = False
        self.encoder.build(input_shape=input_shape)
        self._latent.build(input_shape=input_shape)
        self.decoder.build(input_shape=(None, latent_size))

    @staticmethod
    def _compute_latent(x, training):
        mean, log_var = x
        return mean + K.exp(0.5 * log_var)

    def call(self, inputs, training=None, mask=None):
        inputs = self.encoder(inputs, training, mask)

        inputs = self._flatten(inputs)
        self._mean = self._latent_mean(inputs)
        self._log_var = self._latent_log_var(inputs)

        inputs = self._compute_latent((self._mean, self._log_var), training=training)
        return self.decoder(inputs, training, mask)

    def loss(self, actual, predicted):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(actual, predicted), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + self._log_var - tf.square(self._mean) - tf.exp(self._log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return total_loss

    @classmethod
    def load(cls, filename, latent_size: int = 512, input_shape: tuple = None):
        reloaded = cls(latent_size, input_shape=input_shape)
        reloaded.load_weights(filename)
        if reloaded is not None:
            reloaded.build(input_shape=input_shape)
        return reloaded

    def save(self, filename, fixed: bool = False, **kwargs):
        if fixed:
            super(Generator, self).save(filename, **kwargs)
        else:
            super(Generator, self).save_weights(filename, **kwargs)


def accuracy(y_true, y_pred):
    threshold = .1
    step = tf.abs(tf.subtract(y_true, y_pred)) - (threshold, threshold, threshold)
    step = tf.reduce_all(tf.less(step, (0., 0., 0.)), axis=3)
    good_pixels = tf.where(step, 1.0, 0.0)
    good_pixels = tf.reduce_sum(good_pixels)
    shape = tf.shape(y_pred)
    total_pixels = tf.cast(shape[0] * shape[1] * shape[2], tf.float32)
    return good_pixels / total_pixels


if __name__ == '__main__':
    latent_size = 2048
    input_shape = (None, 64, 64, 3)
    model = Generator(latent_size=latent_size, input_shape=input_shape)
    model.build(input_shape=input_shape)
    model.compile(optimizer=NoisyAdam(strength=.55, sustain=.9, learning_rate=5e-5, beta_1=0.82, beta_2=0.9,
                                      epsilon=0.1),
                  loss=model.loss, metrics=[accuracy, 'mae'])
    model.summary()
    inputs = np.zeros((128, 64, 64, 3), dtype=float)
    outputs = np.ones((128, 64, 64, 3), dtype=float)
    model.fit(x=inputs, y=outputs, batch_size=128, epochs=3)
    model.save_weights('/tmp/model')
    reloaded = Generator.load('/tmp/model', input_shape=(None, 64, 64, 3), latent_size=latent_size)
    print(f"reloaded. type of reloaded is {type(reloaded)}")
    reloaded.compile(optimizer=NoisyAdam(strength=.55, sustain=.9, learning_rate=5e-5, beta_1=0.82, beta_2=0.9),
                     loss=reloaded.loss, metrics=[accuracy, 'mae'])
    reloaded.summary()
    reloaded.fit(x=inputs, y=outputs, batch_size=128, epochs=1)
    y = reloaded(inputs, False)
    print(y.shape)
