from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.initializers.initializers_v2 import RandomNormal
from keras.layers import LeakyReLU, Conv2D, Conv2DTranspose, Normalization, BatchNormalization, UpSampling2D, MaxPool2D, \
    ReLU, Dropout, Flatten, Dense, Reshape, GlobalMaxPooling1D
from keras.models import Sequential


class ImageGenerator:
    def __init__(self):
        self._features = VGG16(include_top=False)
        self._features.trainable = False

        self._generate = Sequential([
            MaxPool2D(2),
            Conv2DTranspose(256, 3, 3, use_bias=False, activation='leaky_relu'),
            Conv2DTranspose(128, 3, 3, use_bias=False, activation='leaky_relu', kernel_regularizer=regularizers.l2()),
            Conv2DTranspose(3, 2, 1, use_bias=False, activation='leaky_relu'),
            Conv2D(256, 4, 2, use_bias=True, activation='leaky_relu'),
            Conv2D(512, 2, 2, use_bias=True, activation='leaky_relu'),
            MaxPool2D(2),
            BatchNormalization(),
            Conv2DTranspose(256, 3, 3, use_bias=False, activation='leaky_relu'),
            Conv2DTranspose(128, 3, 3, use_bias=False, activation='leaky_relu', kernel_regularizer=regularizers.l2()),
            Conv2DTranspose(3, 2, 1, use_bias=False, activation='leaky_relu'),
            Conv2D(256, 4, 2, use_bias=True, activation='leaky_relu'),
            Conv2D(512, 2, 2, use_bias=True, activation='leaky_relu'),
            MaxPool2D(2),
            BatchNormalization(),
            Conv2DTranspose(256, 5, 5, use_bias=False, activation='leaky_relu'),
            Conv2DTranspose(256, 2, 2, use_bias=False, activation='leaky_relu'),
            BatchNormalization(),
            Conv2DTranspose(128, 2, 2, use_bias=True, activation='leaky_relu'),
            Conv2DTranspose(128, 2, 2, use_bias=False, activation='leaky_relu'),
            BatchNormalization(),
            Conv2DTranspose(64, 2, 2, use_bias=True, activation='leaky_relu', kernel_regularizer=regularizers.l2()),
            Conv2DTranspose(3, 1, 1, use_bias=True, activation='sigmoid')
        ])

    def get_model(self) -> Sequential:
        return Sequential([
            self._features,
            self._generate
        ])
