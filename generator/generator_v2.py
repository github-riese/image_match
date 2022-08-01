import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2DTranspose, \
    Lambda, Normalization, GaussianNoise
from keras.losses import binary_crossentropy
from keras.optimizer_v2.nadam import Nadam


class NoisyNadam(Nadam):

    def __init__(self, strength, sustain, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name='Nadam',
                 **kwargs):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, name, **kwargs)
        self.strength = strength
        self.sustain = sustain
        self.epoch = 0

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        if self.strength > 0:
            stddev = self.strength / (self.sustain ** self.epoch)
            layers = len(grads_and_vars)
            grads_and_vars = [
                (tf.add(gradient, tf.random.normal(stddev=stddev * (self.sustain ** (layers - n - 1)), mean=0., shape=gradient.shape)),
                 var) for n, (gradient, var) in enumerate(grads_and_vars)]
        return super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)


class Generator(tf.keras.Model):
    def __init__(self, latent_size: int = 512, input_shape: tuple = (None, 80, 80, 3)):
        super().__init__()

        self._mean = None
        self._log_var = None

        self._norm = Normalization()
        self._noise_1 = GaussianNoise(.15)

        self._vgg = VGG16(include_top=False, input_shape=(None, None, 3), weights='imagenet', pooling='max')
        self._vgg.trainable = False
        self._flatten = Flatten()

        self._latent = Lambda(self._compute_latent, output_shape=(latent_size,))
        self._latent_mean = Dense(latent_size)
        self._latent_log_var = Dense(latent_size)

        self._dense_1 = Dense(latent_size)
        self._dropout = Dropout(rate=0.4)
        self._dense_2 = Dense(latent_size)

        self._reshape = Reshape(target_shape=(1, 1, latent_size))

        self._generate_1 = Conv2DTranspose(512, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_2 = Conv2DTranspose(256, 5, 5, use_bias=False, activation='leaky_relu')
        self._generate_3 = Conv2DTranspose(128, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_4 = Conv2DTranspose(96, 3, 1, use_bias=False, activation='leaky_relu', padding='same')
        self._generate_5 = Conv2DTranspose(72, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_6 = Conv2DTranspose(64, 3, 1, use_bias=False, activation='leaky_relu', padding='same')
        self._generate_7 = Conv2DTranspose(48, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_8 = Conv2DTranspose(48, 2, 1, use_bias=False, activation='leaky_relu', padding='same')
        self._output = Conv2DTranspose(3, 2, 1, use_bias=False, activation='sigmoid', padding='same')

        self._latent.build(input_shape=input_shape)

    @staticmethod
    def _compute_latent(x, training):
        mean, log_var = x
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mean + K.exp(log_var / 2) * epsilon

    def call(self, inputs, training=None, mask=None):
        if training:
            inputs = self._noise_1(inputs)
        inputs = self._norm(inputs)
        inputs = self._vgg(inputs)
        inputs = self._flatten(inputs)
        if training:
            inputs = self._dropout(inputs)

        self._mean = self._latent_mean(inputs)
        self._log_var = self._latent_log_var(inputs)
        inputs = self._compute_latent((self._mean, self._log_var), training)
        if training:
            inputs = self._dropout(inputs)
        inputs = self._dense_1(inputs)
        if training:
            inputs = self._dropout(inputs)

        inputs = self._dense_2(inputs)
        inputs = self._reshape(inputs)  # 1x1xlatent
        inputs = self._generate_1(inputs)  # 2x2x512
        inputs = self._generate_2(inputs)  # 4x4x256

        inputs = self._generate_3(inputs)  # 20x20x128
        inputs = self._generate_4(inputs)  # 20x20x96

        inputs = self._generate_5(inputs)  # 40x40x72
        inputs = self._generate_6(inputs)  # 40x40x64

        inputs = self._generate_7(inputs)  # 80x80x48
        inputs = self._generate_8(inputs)  # 80x80x48
        return self._output(inputs)  # 80x80x3

    def loss(self, actual, predicted):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(actual, predicted), axis=(1, 2)
            )
        )
        # reconstruction_loss = binary_crossentropy(K.flatten(actual), K.flatten(predicted))
        # kl_loss = 1 + self._sigma - K.square(self._mu) - K.exp(self._sigma)
        # kl_loss *= -.5
        kl_loss = -0.2 * (1 + self._log_var - tf.square(self._mean) - tf.exp(self._log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return reconstruction_loss + kl_loss

    @classmethod
    def load(cls, filename, latent_size: int = 512, input_shape: tuple = None):
        reloaded = cls(latent_size)
        reloaded.load_weights(filename)
        if reloaded is not None:
            reloaded.build(input_shape=input_shape)
        return reloaded

    def save(self, filename, **kwargs):
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
    model = Generator(latent_size=latent_size)
    model.build(input_shape=(None, 80, 80, 3))
    model.compile(optimizer=NoisyNadam(strength=.55, sustain=.9, learning_rate=5e-5, beta_1=0.82, beta_2=0.9),
                  loss=model.loss, metrics=[accuracy, 'mae'])
    model.summary()
    inputs = np.zeros((32, 80, 80, 3), dtype=float)
    outputs = np.ones((32, 80, 80, 3), dtype=float)
    model.fit(x=inputs, y=outputs, epochs=3)
    model.save_weights('/tmp/model')
    reloaded = Generator.load('/tmp/model', input_shape=(None, 80, 80, 3), latent_size=latent_size)
    print(f"reloaded. type of reloaded is {type(reloaded)}")
    reloaded.compile(optimizer=NoisyNadam(strength=.55, sustain=.9, learning_rate=5e-5, beta_1=0.82, beta_2=0.9),
                     loss=reloaded.loss, metrics=[accuracy, 'mae'])
    reloaded.summary()
    reloaded.fit(x=inputs, y=outputs, epochs=1)
    y = reloaded.call(inputs, False)
    print(y.shape)
