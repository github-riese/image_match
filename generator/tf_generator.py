from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from keras.models import Sequential


class ImageGenerator:
    def __init__(self):
        self._features = VGG16(include_top=False)
        self._features.trainable = False

        self._generate = Sequential([
            BatchNormalization(),
            Conv2DTranspose(128, 4, 4, activation=LeakyReLU(0.2), padding='same'),
            Conv2DTranspose(64, 2, 2, activation=LeakyReLU(0.2)),
            Conv2D(64, 4, 2),
            Conv2DTranspose(32, 4, 2, activation=LeakyReLU(0.2)),
            Conv2DTranspose(32, 5, 5, activation=LeakyReLU(0.2)),
            BatchNormalization(),
            Conv2DTranspose(16, 1, 1, activation=LeakyReLU(0.2)),
            Conv2DTranspose(3, 1, 1, activation='sigmoid')
        ])

    def get_model(self) -> Sequential:
        return Sequential([
            self._features,
            self._generate
        ])
