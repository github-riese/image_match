import torch.nn
from torch.nn import Module, init, Sequential, MaxPool2d, Conv2d, LeakyReLU, BatchNorm2d, Flatten, Linear, Upsample, \
    Sigmoid


class ImageGenerator(Module):
    _features: Module
    _generate: Module

    def __init__(self, vgg16_features: Module):
        super(ImageGenerator, self).__init__()
        self._features = vgg16_features
        self.disable_learning(self._features)

        self._generate = Sequential(
            BatchNorm2d(512),
            Upsample(scale_factor=8),
            Conv2d(512, 128, 4, 1, 1),
            Conv2d(128, 128, 2, 2, 1),
            LeakyReLU(inplace=True),
            Upsample(scale_factor=8),
            Conv2d(128, 128, 4, 2),
            Conv2d(128, 128, 2, 2),
            LeakyReLU(inplace=True),
            Upsample(size=(80, 80)),
            Conv2d(128, 3, 1, 1, 0),
            Sigmoid()
        )

    def forward(self, X):
        X = self._features.forward(X)
        return self._generate.forward(X)

    def init_weights(self):
        def _init(m):
            class_name = m.__class__.__name__
            if class_name in ['Conv2d', 'ConvTranspose2d']:
                init.normal_(m.weight.data, .0, .02)
            elif class_name in ['BatchNorm', 'BatchNorm2d']:
                init.normal_(m.weight.data, 1.0, .02)
                init.constant_(m.bias.data, 0)
            if class_name == 'Linear':
                init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self._detect.apply(_init)

    @staticmethod
    def disable_learning(module: Module):
        for param in module.parameters():
            param.requires_grad = False
