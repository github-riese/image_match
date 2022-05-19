import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.callbacks import Callback
from keras.losses import BinaryCrossentropy
from keras.optimizer_v2.nadam import Nadam
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.utils.data import DataLoader

from augmented_image_loader import AugmentedImageLoader
from generator.dataset import Dataset
from generator.tf_generator import ImageGenerator
from image_display import ImageDisplay
from image_loader import ImageLoader
from tf_dataloader_wrapper import dataloader_wrapper


class PlottingCallback(Callback):

    def __init__(self, display: ImageDisplay, sources: np.ndarray, validate: tf.Tensor, expect: np.ndarray):
        super(PlottingCallback, self).__init__()
        self._display = display
        self._sources = sources
        self._validate = validate
        self._expect = expect

    def on_epoch_end(self, epoch, logs=None):
        bamm = self.model.predict(tf.convert_to_tensor(self._validate, dtype=tf.float32))
        bamm = np.concatenate([self._sources, bamm, self._expect], axis=0)
        self._display.show_images(bamm, 5)
        self._display.save(f"snapshots/image_ep_{epoch + 1:03d}_{logs['loss']:.4f}.png")
        if epoch % 5 == 4:
            self.model.save("models/tf_model.tf")
            print("model saved.")
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

    if os.path.exists(model_filename):
        model = tf.keras.models.load_model(model_filename)
    else:
        model = ImageGenerator().get_model()
    model.compile(optimizer=Nadam(learning_rate=0.0001, beta_1=0.9),
                  loss=BinaryCrossentropy())
    model.summary()

    #    validation_ids = list(enumerate(validation_dataset))
    #    random.shuffle(validation_ids)
    validation_ids = [2, 13, 21, 37, 53]

    sources = np.zeros((0, 80, 80, 3))
    validate = tf.zeros((0, 80, 80, 3))
    expect = np.zeros((0, 80, 80, 3))

    for n in range(5):
        index = validation_ids[n]
        x, y = validation_dataset.__getitem__(index)
        validate = tf.concat([validate, x.reshape((1, 80, 80, 3))], axis=0)
        sources = np.concatenate([sources, x.reshape((1, 80, 80, 3))], axis=0)
        expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)
    display = ImageDisplay()

    batch_size = 30
    epochs = 1000

    callback = PlottingCallback(display, sources, validate, expect)

    model.fit(dataloader_wrapper(dataset, True, batch_size),
              steps_per_epoch=len(dataset) / batch_size,
              epochs=epochs, validation_freq=1, verbose=1,
              callbacks=callback, initial_epoch=200)
    model.save(model_filename)
    plt.waitforbuttonpress()


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
        'csv_file': args[3]
    }
