from keras.applications.vgg16 import VGG16
from keras.initializers.initializers_v2 import RandomNormal
from keras.layers import LeakyReLU, Conv2D, Conv2DTranspose, Normalization, BatchNormalization, UpSampling2D
from keras.models import Sequential


class ImageGenerator:
    def __init__(self):
        self._features = VGG16(include_top=False)
        self._features.trainable = False

        self._generate = Sequential([
            Conv2DTranspose(256, 10, 5, activation=LeakyReLU(), padding='same', kernel_initializer=RandomNormal(),
                            use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(256, 4, 4, activation=LeakyReLU(), kernel_initializer=RandomNormal(), use_bias=False),
            Conv2DTranspose(128, 2, 2, activation=LeakyReLU(), kernel_initializer=RandomNormal(), use_bias=False),
            Conv2DTranspose(128, 1, 1, activation=LeakyReLU(), use_bias=False, kernel_initializer=RandomNormal()),
            BatchNormalization(),
            Conv2D(3, 1, 1, use_bias=False, activation='sigmoid', kernel_initializer=RandomNormal())
        ])

    def get_model(self) -> Sequential:
        return Sequential([
            self._features,
            self._generate
        ])
