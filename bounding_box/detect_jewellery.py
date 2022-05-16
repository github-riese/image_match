from torch.nn import Module, Sequential, Linear, Sigmoid, init, Flatten, MaxPool2d, BatchNorm2d, LeakyReLU


class DetectJewellery(Module):
    def __init__(self, vgg16_features: Module):
        super(DetectJewellery, self).__init__()
        self._features = vgg16_features
        self.disable_learning(self._features)
        self._detect = Sequential(
            BatchNorm2d(512),
            MaxPool2d(kernel_size=4, stride=4, padding=2),
            Flatten(),
            Linear(2048, 128),
            LeakyReLU(inplace=True),
            Linear(128, 64),
            LeakyReLU(inplace=True),
            Linear(64, 32),
            LeakyReLU(inplace=True),
            Linear(32, 4),
            Sigmoid()
        )

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

    def forward(self, X):
        X = self._features.forward(X)
        return self._detect.forward(X)
