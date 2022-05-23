import os
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.callbacks import Callback
from keras.losses import BinaryCrossentropy, LogCosh, CategoricalCrossentropy, CosineSimilarity, MeanAbsoluteError
from keras.optimizer_v2.nadam import Nadam
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.utils.data import DataLoader

from augmented_image_loader import AugmentedImageLoader
from generator.dataset import Dataset
from generator.tf_generator import ImageGenerator
from image_display import ImageDisplay
from image_loader import ImageLoader
from tf_dataloader_wrapper import dataloader_wrapper, normalize_x


class PlottingCallback(Callback):

    def __init__(self, display: ImageDisplay, sources: np.ndarray, validate: tf.Tensor, expect: np.ndarray,
                 loss_fd: int):
        super(PlottingCallback, self).__init__()
        self._display = display
        self._sources = sources
        self._validate = validate
        self._expect = expect
        self._loss_fd = loss_fd

    def on_epoch_end(self, epoch, logs=None):
        bamm = self.model.predict(tf.convert_to_tensor(self._validate, dtype=tf.float32))
        bamm = np.concatenate([self._sources, bamm, self._expect], axis=0)
        self._display.show_images(bamm, self._sources.shape[0])
        self._display.save(f"snapshots/image_ep_{epoch + 1:03d}_{logs['loss']:.4f}.png")
        if epoch % 5 == 4:
            self.model.save("models/tf_model.tf")
            print("model saved.")
        os.write(self._loss_fd, bytes(f"{epoch + 1}, {logs['loss']:.5f}\n", "UTF-8"))
        return super().on_epoch_end(epoch, logs)


def generate_default_view(args: list):
    config = _configure(args)
    base_image_loader = ImageLoader({'jewellery': config['image_path']}, (240, 240))
    data = pd.read_csv(config['csv_file'])
    image_loader = AugmentedImageLoader(image_loader=base_image_loader, size=(80, 80))
    dataset = Dataset(data, reserve_percent=.925, image_loader=image_loader)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True)
    validation_dataset = dataset.get_reserved_data()
    validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False)

    model_filename = "models/tf_model.tf"

    if os.path.exists("loss.csv"):
        losses = os.open("loss.csv", os.O_WRONLY | os.O_APPEND)
    else:
        losses = os.open("loss.csv", os.O_WRONLY | os.O_CREAT)
        os.write(losses, bytes("epoch,loss\n", "UTF-8"))

    if os.path.exists(model_filename):
        model = tf.keras.models.load_model(model_filename)
    else:
        model = ImageGenerator().get_model()
    model.compile(optimizer=Nadam(learning_rate=0.0001, beta_1=0.95, beta_2=0.999),
                  loss=MeanAbsoluteError())
    model.summary()

    #    validation_ids = list(enumerate(validation_dataset))
    #    random.shuffle(validation_ids)
    validation_ids = [2, 13, 21, 37, 53]

    expect, sources, validate = make_validation_data(validation_dataset, validation_ids)
    x = image_loader.load_image('/Users/riese/tmp/images/3343OO_screenshot.jpg', desired_size=(80, 80))
    y = image_loader.load_image('/Users/riese/tmp/images/3343OO.png', desired_size=(80, 80))
    x = np.array(x / 255, dtype=np.float32)
    y = np.array(y / 255, dtype=np.float32)
    validate = tf.concat([validate, np.reshape(normalize_x(deepcopy(x)), (1, 80, 80, 3))], axis=0)
    sources = np.concatenate([sources, x.reshape((1, 80, 80, 3))], axis=0)
    expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)

    display = ImageDisplay()

    batch_size = 50
    epochs = config['epochs']

    callback = PlottingCallback(display, sources, validate, expect, losses)

    model.fit(dataloader_wrapper(dataset, True, batch_size),
              steps_per_epoch=len(dataset) / batch_size,
              epochs=epochs, validation_freq=1, verbose=1,
              validation_data=(sources, expect),
              callbacks=callback, initial_epoch=config['initial_epoch'])
    model.save(model_filename)
    os.close(losses)
    plt.waitforbuttonpress()


def make_validation_data(validation_dataset, validation_ids):
    sources = np.zeros((0, 80, 80, 3))
    validate = tf.zeros((0, 80, 80, 3))
    expect = np.zeros((0, 80, 80, 3))
    for n in range(5):
        index = validation_ids[n]
        x, y = validation_dataset.__getitem__(index)
        validate = tf.concat([validate, normalize_x(deepcopy(x)).reshape((1, 80, 80, 3))], axis=0)
        sources = np.concatenate([sources, x.reshape((1, 80, 80, 3))], axis=0)
        expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)
    return expect, sources, validate


def load_model() -> Module:
    file_name = os.path.realpath('.') + '/models/GenerateImage.pth'
    if os.path.exists(file_name):
        return torch.load(file_name)
    vgg_file = os.path.realpath('.') + '/models/VGG16-pretrained-no-classification.pth'
    vgg = torch.load(vgg_file)
    image_generator = ImageGenerator(vgg)
    return image_generator


def _configure(args: list) -> dict:
    return {
        'model': args[1],
        'image_path': args[2],
        'csv_file': args[3],
        'epochs': int(args[4] if len(args) > 4 else 1000),
        'initial_epoch': int(args[5] if len(args) > 5 else 0),
    }
