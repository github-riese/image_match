from keras.applications.vgg16 import VGG16
from keras.initializers.initializers_v2 import RandomNormal
from keras.layers import LeakyReLU, Conv2D, Conv2DTranspose, Normalization, BatchNormalization, UpSampling2D, MaxPool2D, \
    ReLU, Dropout
from keras.models import Sequential


class ImageGenerator:
    def __init__(self):
        self._features = VGG16(include_top=False)
        self._features.trainable = False

        self._generate = Sequential([
            Conv2DTranspose(256, 4, 4, use_bias=False),
            Conv2D(128, 4, 2, use_bias=False, activation='selu'),
            MaxPool2D(),
            Conv2DTranspose(128, 5, 5, use_bias=False),
            Conv2DTranspose(128, 2, 2, use_bias=False, activation='selu'),
            Conv2DTranspose(64, 4, 2, use_bias=False, padding='same'),
            Conv2DTranspose(32, 4, 2, use_bias=False, padding='same'),
            Conv2DTranspose(16, 2, 2, use_bias=False, activation='selu'),
            Conv2DTranspose(3, 1, 1, use_bias=False, activation='sigmoid')
        ])

    def get_model(self) -> Sequential:
        return Sequential([
            self._features,
            self._generate
        ])
