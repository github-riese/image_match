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
            MaxPool2D(2),
            Conv2DTranspose(256, 3, 3, use_bias=False, activation='leaky_relu'),
            BatchNormalization(),
            Conv2DTranspose(128, 3, 3, use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(64, 2, 1, use_bias=True, activation='leaky_relu'),
            BatchNormalization(),
            Dropout(rate=0.2),
            Conv2D(256, 4, 2),
            BatchNormalization(),
            Conv2D(512, 2, 2),
            BatchNormalization(),
            MaxPool2D(2),
            Conv2DTranspose(512, 5, 5, use_bias=True),
            BatchNormalization(),
            Dropout(rate=0.33),
            Conv2DTranspose(256, 2, 2, use_bias=False, activation='leaky_relu'),
            BatchNormalization(),
            Conv2DTranspose(128, 2, 2, use_bias=False, activation='leaky_relu'),
            BatchNormalization(),
            Conv2DTranspose(64, 2, 2, use_bias=False, activation='leaky_relu'),
            BatchNormalization(),
            Conv2DTranspose(32, 2, 2, use_bias=False, activation='leaky_relu'),
            BatchNormalization(),
            Conv2DTranspose(3, 1, 1, use_bias=False, activation='sigmoid')
        ])

    def get_model(self) -> Sequential:
        return Sequential([
            self._features,
            self._generate
        ])
