from tensorflow import keras
import math

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2DTranspose, BatchNormalization, \
    Lambda, Normalization
from keras.losses import binary_crossentropy
from keras.optimizer_v2.adam import Adam


class Generator(tf.keras.Model):
    def __init__(self, latent_size: int = 512, input_shape: tuple = (None, 80, 80, 3)):
        super().__init__()

        self._mu = None
        self._sigma = None

        self._norm = Normalization()
        self._vgg = VGG16(include_top=False, input_shape=(None, None, 3), weights='imagenet')
        self._vgg.trainable = False
        self._flatten = Flatten()

        self._latent = Lambda(self._compute_latent, output_shape=(latent_size,))
        self._latent_mu = Dense(latent_size)
        self._latent_sigma = Dense(latent_size)

        self._dense1 = Dense(latent_size, activation='leaky_relu')
        self._dropout = Dropout(rate=0.4)
        self._dense2 = Dense(latent_size, activation='leaky_relu')
        self._dense3 = Dense(latent_size, activation='leaky_relu')

        self._reshape = Reshape(target_shape=(1, 1, latent_size))

        self._generate_1 = Conv2DTranspose(latent_size, 5, 5, use_bias=False, activation='leaky_relu',
                                           kernel_regularizer=regularizers.l2(0.001))
        self._batch_norm_1 = BatchNormalization()
        self._generate_2 = Conv2DTranspose(int(math.ceil(latent_size / 2)), 2, 2, use_bias=False,
                                           activation='leaky_relu')
        self._generate_3 = Conv2DTranspose(int(math.ceil(latent_size / 2)), 2, 2, use_bias=False,
                                           activation='leaky_relu',
                                           kernel_regularizer=regularizers.l2(0.002))
        self._batch_norm_2 = BatchNormalization()
        self._generate_4 = Conv2DTranspose(128, 2, 2, use_bias=False, activation='leaky_relu',
                                           kernel_regularizer=regularizers.l2(0.005))
        self._generate_5 = Conv2DTranspose(128, 2, 2, use_bias=False, activation='leaky_relu',
                                           kernel_regularizer=regularizers.l2())
        self._generate_6 = Conv2DTranspose(3, 1, 1, use_bias=True, activation='sigmoid')

        self._latent.build(input_shape=input_shape)

    @staticmethod
    def _compute_latent(x):
        mu, sigma = x
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mu + K.exp(sigma / 2) * epsilon

    def call(self, inputs, training=None, mask=None):
        inputs = self._norm(inputs)
        inputs = self._vgg(inputs)
        inputs = self._flatten(inputs)
        self._mu = self._latent_mu(inputs)
        self._sigma = self._latent_sigma(inputs)
        inputs = self._compute_latent((self._mu, self._sigma))
        inputs = self._dense1(inputs)
        if training:
            inputs = self._dropout(inputs)
        inputs = self._dense2(inputs)
        if training:
            inputs = self._dropout(inputs)
        inputs = self._dense3(inputs)
        inputs = self._reshape(inputs)
        inputs = self._generate_1(inputs)
        inputs = self._batch_norm_1(inputs)
        inputs = self._dropout(inputs)
        inputs = self._generate_2(inputs)
        inputs = self._generate_3(inputs)
        inputs = self._batch_norm_2(inputs)
        inputs = self._generate_4(inputs)
        inputs = self._generate_5(inputs)
        return self._generate_6(inputs)

    def loss(self, actual, predicted):
        reconstruction_loss = binary_crossentropy(K.flatten(actual), K.flatten(predicted)) * 80 * 80 * 3
        kl_loss = 1 + self._sigma - K.square(self._mu) - K.exp(self._sigma)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    @classmethod
    def load(cls, filename, latent_size: int = 512, input_shape: tuple = None):
        reloaded = cls(latent_size)
        reloaded.load_weights(filename)
        if reloaded is not None:
            reloaded.build(input_shape=input_shape)
        return reloaded

    def save(self, filename, **kwargs):
        super(Generator, self).save_weights(filename, **kwargs)


if __name__ == '__main__':
    model = Generator()
    model.build(input_shape=(None, 80, 80, 3))
    model.compile(optimizer=Adam(learning_rate=2.5e-6, beta_1=0.5, beta_2=0.75), loss=model.loss)
    model.summary()
    inputs = np.zeros((32, 80, 80, 3), dtype=float)
    outputs = np.ones((32, 80, 80, 3), dtype=float)
    model.fit(x=inputs, y=outputs, epochs=1)
    model.save_weights('/tmp/model')
    reloaded = Generator.load('/tmp/model', input_shape=(None, 80, 80, 3))
    print(f"reloaded. type of reloaded is {type(reloaded)}")
    reloaded.compile(optimizer=Adam(learning_rate=2.5e-6, beta_1=0.5, beta_2=0.75), loss=reloaded.loss)
    reloaded.summary()
    reloaded.fit(x=inputs, y=outputs, epochs=1)
    y = reloaded.call(inputs, False)
    print(y.shape)
