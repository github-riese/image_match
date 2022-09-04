import numpy as np
import tensorflow as tf
from keras import backend as K, regularizers
from keras.backend import flatten
from keras.layers import Dense, Flatten, Reshape, Conv2DTranspose, \
    Lambda, Dropout, GaussianNoise, MaxPooling2D, Normalization
from keras.legacy_tf_layers.convolutional import Conv2D
from keras.optimizers import Nadam


class NoisyAdam(Nadam):

    def __init__(self, strength, sustain, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name='NoisyAdam',
                 **kwargs):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, name, **kwargs)
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
                           tf.random.truncated_normal(stddev=stddev * .9 ** (layers - n), mean=0., shape=gradient.shape)),
                    var) for n, (gradient, var) in enumerate(grads_and_vars)]
        return super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)


class Generator(tf.keras.Model):
    def __init__(self, latent_size: int = 512, input_shape: tuple = (None, 80, 80, 3)):
        super().__init__()

        self._mean = None
        self._log_var = None

        self._norm = Normalization()
        self._noise_1 = GaussianNoise(.5)
        self._noise_2 = GaussianNoise(.45)
        self._noise_3 = GaussianNoise(.405)
        self._noise_4 = GaussianNoise(.3645)

        self._conv_1 = Conv2D(64, 3, activation='leaky_relu', padding='same')
        self._conv_2 = Conv2D(64, 3, activation='leaky_relu', padding='same')
        self._pool_1 = MaxPooling2D(3, 3)

        self._conv_3 = Conv2D(128, 3, activation='leaky_relu', padding='same')
        self._conv_4 = Conv2D(128, 3, activation='leaky_relu', padding='same')
        self._pool_2 = MaxPooling2D(3, 3)

        self._conv_5 = Conv2D(256, 3, activation='leaky_relu', padding='same')
        self._conv_6 = Conv2D(256, 3, activation='leaky_relu', padding='same')
        self._pool_3 = MaxPooling2D(3, 3)

        self._conv_7 = Conv2D(512, 3, activation='leaky_relu', padding='same')
        self._conv_8 = Conv2D(512, 3, activation='leaky_relu', padding='same')
        self._pool_4 = MaxPooling2D(2, 2)

        self._flatten = Flatten()

        self._dropout = Dropout(rate=0.4)
        self._latent = Lambda(self._compute_latent, output_shape=(latent_size,))
        self._latent_mean = Dense(latent_size)
        self._latent_log_var = Dense(latent_size)

        self._dense_1 = Dense(latent_size, activation='leaky_relu')
        self._dense_2 = Dense(latent_size, activation='leaky_relu')

        self._reshape = Reshape(target_shape=(1, 1, latent_size))

        self._generate_1 = Conv2DTranspose(512, 2, 2, use_bias=True, activation='leaky_relu')
        self._generate_2 = Conv2DTranspose(256, 2, 2, use_bias=True, activation='leaky_relu')
        self._generate_3 = Conv2DTranspose(128, 2, 2, use_bias=True, activation='leaky_relu')
        self._generate_4 = Conv2DTranspose(96, 3, 1, use_bias=True, activation='leaky_relu', padding='same')
        self._generate_5 = Conv2DTranspose(72, 2, 2, use_bias=True, activation='leaky_relu')
        self._generate_6 = Conv2DTranspose(64, 3, 1, use_bias=True, activation='leaky_relu', padding='same')
        self._generate_7 = Conv2DTranspose(48, 2, 2, use_bias=True, activation='leaky_relu')
        self._generate_8 = Conv2DTranspose(48, 2, 1, use_bias=True, activation='leaky_relu', padding='same')
        self._output = Conv2DTranspose(3, 2, 2, use_bias=True, activation='sigmoid')

        self._latent.build(input_shape=input_shape)

    @staticmethod
    def _compute_latent(x, training):
        mean, log_var = x
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mean + K.exp(log_var / 2) * epsilon

    def call(self, inputs, training=None, mask=None):
        inputs = self._norm(inputs)
        if training:
            inputs = self._noise_1(inputs)
        inputs = self._conv_1(inputs)
        inputs = self._conv_2(inputs)
        inputs = self._pool_1(inputs)
        if training:
            inputs = self._noise_2(inputs)
        inputs = self._conv_3(inputs)
        inputs = self._conv_4(inputs)
        inputs = self._pool_2(inputs)
        if training:
            inputs = self._noise_3(inputs)
        inputs = self._conv_5(inputs)
        inputs = self._conv_6(inputs)
        inputs = self._pool_3(inputs)
        if training:
            inputs = self._noise_4(inputs)
        inputs = self._conv_7(inputs)
        inputs = self._conv_8(inputs)
        inputs = self._pool_4(inputs)

        inputs = self._flatten(inputs)

        if training:
            inputs = self._dropout(inputs)
        inputs = self._dense_1(inputs)
        if training:
            inputs = self._dropout(inputs)
        inputs = self._dense_2(inputs)

        if training:
            inputs = self._dropout(inputs)
        self._mean = self._latent_mean(inputs)
        self._log_var = self._latent_log_var(inputs)

        inputs = self._compute_latent((self._mean, self._log_var), training=training)

        inputs = self._reshape(inputs)  # 1x1xlatent
        inputs = self._generate_1(inputs)  # 2x2x512
        inputs = self._generate_2(inputs)  # 4x4x256
        inputs = self._generate_3(inputs)  # 8x8x128
        inputs = self._generate_4(inputs)  # 8x8x96
        inputs = self._generate_5(inputs)  # 16x16x72
        inputs = self._generate_6(inputs)  # 16x16x64
        inputs = self._generate_7(inputs)  # 32x32x48
        inputs = self._generate_8(inputs)  # 32x32x48
        return self._output(inputs)  # 64x64x3

    def loss(self, actual, predicted):
        reconstruction_loss = tf.keras.losses.binary_crossentropy(flatten(actual), flatten(predicted)) * 64 * 64 * 3
        kl_loss = -0.5 * (1 + self._log_var - tf.square(self._mean) - tf.exp(self._log_var))
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    @classmethod
    def load(cls, filename, latent_size: int = 512, input_shape: tuple = None):
        reloaded = cls(latent_size)
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
    latent_size = 512
    model = Generator(latent_size=latent_size)
    model.build(input_shape=(None, 64, 64, 3))
    model.compile(optimizer=NoisyAdam(strength=.55, sustain=.9, learning_rate=5e-5, beta_1=0.82, beta_2=0.9),
                  loss=model.loss, metrics=[accuracy, 'mae'])
    model.summary()
    inputs = np.zeros((32, 64, 64, 3), dtype=float)
    outputs = np.ones((32, 64, 64, 3), dtype=float)
    model.fit(x=inputs, y=outputs, epochs=3)
    model.save_weights('/tmp/model')
    reloaded = Generator.load('/tmp/model', input_shape=(None, 64, 64, 3), latent_size=latent_size)
    print(f"reloaded. type of reloaded is {type(reloaded)}")
    reloaded.compile(optimizer=NoisyNadam(strength=.55, sustain=.9, learning_rate=5e-5, beta_1=0.82, beta_2=0.9),
                     loss=reloaded.loss, metrics=[accuracy, 'mae'])
    reloaded.summary()
    reloaded.fit(x=inputs, y=outputs, epochs=1)
    y = reloaded(inputs, False)
    print(y.shape)
