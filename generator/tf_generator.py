from keras.applications.vgg16 import VGG16
from keras.initializers.initializers_v2 import RandomNormal
from keras.layers import BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose, UpSampling2D
from keras.models import Sequential


class ImageGenerator:
    def __init__(self):
        self._features = VGG16(include_top=False)
        self._features.trainable = False

        self._generate = Sequential([
            Conv2DTranspose(255, 10, 5, activation=LeakyReLU(0.2), padding='same',
                            kernel_initializer=RandomNormal(mean=0.1, stddev=0.1),
                            bias_initializer='zeros'),
            Conv2DTranspose(256, 4, 4, activation=LeakyReLU(0.1), kernel_initializer=RandomNormal(mean=0.1, stddev=0.1),
                            bias_initializer='zeros'),
            Conv2DTranspose(128, 2, 2, activation=LeakyReLU(0.2), kernel_initializer=RandomNormal(mean=0.1, stddev=0.1),
                            bias_initializer='zeros'),
            Conv2DTranspose(128, 2, 2, activation=LeakyReLU(0.2), kernel_initializer=RandomNormal(mean=0.1, stddev=0.1),
                            bias_initializer='zeros'),
            Conv2D(3, 2, 2, activation='sigmoid', kernel_initializer=RandomNormal(.001))
        ])

    def get_model(self) -> Sequential:
        return Sequential([
            self._features,
            self._generate
        ])
