import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2DTranspose, \
    Lambda, Normalization, GaussianNoise
from keras.losses import binary_crossentropy
from keras.optimizer_v2.adam import Adam


class Generator(tf.keras.Model):
    def __init__(self, latent_size: int = 512, input_shape: tuple = (None, 80, 80, 3)):
        super().__init__()

        self._mu = None
        self._sigma = None

        self._norm = Normalization()
        self._noise_1 = GaussianNoise(.15)
        self._noise_2 = GaussianNoise(.1)
        self._noise_3 = GaussianNoise(.05)

        self._vgg = VGG16(include_top=False, input_shape=(None, None, 3), weights='imagenet')
        self._vgg.trainable = False
        self._flatten = Flatten()

        self._latent = Lambda(self._compute_latent, output_shape=(latent_size,))
        self._latent_mu = Dense(latent_size)
        self._latent_sigma = Dense(latent_size)

        self._dropout = Dropout(rate=0.4)

        self._dense = Dense(1024)
        self._reshape = Reshape(target_shape=(1, 1, 1024))

        self._generate_1 = Conv2DTranspose(512, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_2 = Conv2DTranspose(512, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_3 = Conv2DTranspose(256, 5, 5, use_bias=False, activation='leaky_relu')
        self._generate_4 = Conv2DTranspose(256, 2, 1, use_bias=False, activation='leaky_relu', padding='same')
        self._generate_5 = Conv2DTranspose(128, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_6 = Conv2DTranspose(128, 2, 1, use_bias=False, activation='leaky_relu', padding='same')
        self._generate_7 = Conv2DTranspose(72, 2, 2, use_bias=False, activation='leaky_relu')
        self._generate_8 = Conv2DTranspose(72, 2, 1, use_bias=False, activation='leaky_relu', padding='same')
        self._output = Conv2DTranspose(3, 2, 1, use_bias=False, activation='sigmoid', padding='same')

        self._latent.build(input_shape=input_shape)

    @staticmethod
    def _compute_latent(x, training):
        mu, sigma = x
        if training:
            batch = K.shape(mu)[0]
            dim = K.int_shape(mu)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return mu + K.exp(sigma / 2) * epsilon
        else:
            return mu + K.exp(sigma / 2)

    def call(self, inputs, training=None, mask=None):
        inputs = self._norm(inputs)
        if training:
            inputs = self._noise_1(inputs)
        inputs = self._vgg(inputs)
        inputs = self._flatten(inputs)
        if training:
            inputs = self._dropout(inputs)

        self._mu = self._latent_mu(inputs)
        self._sigma = self._latent_sigma(inputs)
        inputs = self._compute_latent((self._mu, self._sigma), training)
        if training:
            inputs = self._dropout(inputs)
        inputs = self._dense(inputs)
        if training:
            inputs = self._noise_2(inputs)
        inputs = self._reshape(inputs)  # 1x1x1024
        inputs = self._generate_1(inputs)  # 2x2x512
        inputs = self._generate_2(inputs)  # 4x4x512
        if training:
            inputs = self._noise_3(inputs)
        inputs = self._generate_3(inputs)  # 20x20x256
        inputs = self._generate_4(inputs)  # 20x20x256
        if training:
            inputs = self._noise_3(inputs)
        inputs = self._generate_5(inputs)  # 40x40x128
        inputs = self._generate_6(inputs)  # 40x40x128
        if training:
            inputs = self._noise_3(inputs)
        inputs = self._generate_7(inputs)  # 80x80x72
        inputs = self._generate_8(inputs)  # 80x80x72
        return self._output(inputs)  # 80x80x3

    def loss(self, actual, predicted):
        reconstruction_loss = binary_crossentropy(K.flatten(actual), K.flatten(predicted))
        kl_loss = 1 + self._sigma - K.square(self._mu) - K.exp(self._sigma)
        kl_loss *= -.5
        return reconstruction_loss * 16 + kl_loss * 4

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
    threshold = .05
    shape = K.shape(y_true)
    diff = tf.where(K.abs(tf.subtract(y_true, y_pred) - threshold) > 0, 1.0, 0.0)
    total_pixels = tf.cast(tf.reduce_prod(shape), tf.float32)
    return 1.0 - K.sum(diff) / total_pixels


if __name__ == '__main__':
    latent_size = 1152
    model = Generator(latent_size=latent_size)
    model.build(input_shape=(None, 80, 80, 3))
    model.compile(optimizer=Adam(learning_rate=5e-5, beta_1=0.82, beta_2=0.9),
                  loss=model.loss, metrics=['accuracy', 'mae', 'cosine_similarity'])
    model.summary()
    inputs = np.zeros((32, 80, 80, 3), dtype=float)
    outputs = np.ones((32, 80, 80, 3), dtype=float)
    model.fit(x=inputs, y=outputs, epochs=1)
    model.save_weights('/tmp/model')
    reloaded = Generator.load('/tmp/model', input_shape=(None, 80, 80, 3), latent_size=latent_size)
    print(f"reloaded. type of reloaded is {type(reloaded)}")
    reloaded.compile(optimizer=Adam(learning_rate=5e-5, beta_1=0.82, beta_2=0.9),
                     loss=reloaded.loss, metrics=['accuracy', 'mae', 'cosine_similarity'])
    reloaded.summary()
    reloaded.fit(x=inputs, y=outputs, epochs=1)
    y = reloaded.call(inputs, False)
    print(y.shape)
