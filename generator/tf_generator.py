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
            MaxPool2D(pool_size=(2, 2)),
            Conv2DTranspose(256, 4, 4, use_bias=False),
            Conv2DTranspose(256, 4, 4, use_bias=False),
            Conv2DTranspose(256, 2, 2, use_bias=True, activation='swish'),
            BatchNormalization(),
            Conv2D(128, 3, 2, use_bias=False),
            Conv2D(256, 3, 1, use_bias=False),
            Conv2D(512, 2, 2, use_bias=True, activation='swish'),
            MaxPool2D(pool_size=(2, 2)),
            Conv2DTranspose(256, 4, 4, use_bias=False),
            Conv2DTranspose(256, 2, 2, use_bias=False),
            Conv2DTranspose(256, 2, 2, use_bias=True, activation='swish'),
            BatchNormalization(),
            Conv2D(128, 3, 2, use_bias=False),
            Conv2D(256, 3, 2, use_bias=False),
            Conv2D(512, 2, 2, use_bias=True, activation='swish'),
            MaxPool2D(pool_size=(3, 3)),
            Conv2DTranspose(256, 5, 5, use_bias=False),
            Conv2DTranspose(256, 2, 2, use_bias=False, activation='swish'),
            Conv2DTranspose(256, 2, 2, use_bias=False, padding='same', activation='swish'),
            Conv2DTranspose(256, 2, 2, use_bias=False, padding='same', activation='swish'),
            BatchNormalization(),
            Conv2DTranspose(384, 2, 2, use_bias=True, activation='leaky_relu'),
            Conv2DTranspose(3, 1, 1, use_bias=False, activation='sigmoid')
        ])

    def get_model(self) -> Sequential:
        return Sequential([
            self._features,
            self._generate
        ])
