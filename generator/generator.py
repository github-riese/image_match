import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.callbacks import Callback
from keras.losses import BinaryCrossentropy, LogCosh, CategoricalCrossentropy, CosineSimilarity, MeanAbsoluteError, \
    MeanSquaredError, Huber
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.gradient_descent import SGD
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
        self._display.show_images(bamm, self._sources.shape[0], losses=(logs['loss'], logs['val_loss']))
        self._display.save(f"snapshots/image_ep_{epoch + 1:03d}_{logs['loss']:.4f}.png")
        if epoch % 5 == 4:
            self.model.save("models/tf_model.tf")
            print("model saved.")
        os.write(self._loss_fd, bytes(
            f"{epoch + 1}, {logs['loss']:.6f}, {logs['val_loss']:.6f}, "
            f"{logs['accuracy']:.6f}, {logs['val_accuracy']:.6f}\n",
            "UTF-8"))
        return super().on_epoch_end(epoch, logs)


def generate_default_view(args: list):
    config = _configure(args)
    base_image_loader = ImageLoader({'jewellery': config['image_path']}, (240, 240),
                                    cache_path=config['image_path'] + "/cache")
    data = pd.read_csv(config['csv_file'])
    image_loader = AugmentedImageLoader(image_loader=base_image_loader, size=(80, 80))
    dataset = Dataset(data, reserve_percent=.925, image_loader=image_loader)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True)
    validation_dataset = dataset.get_reserved_data()
    validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False)

    print(f"train dataset contains {len(dataset)} samples, validation up to {len(validation_dataset)}.")

    model_filename = "models/tf_model.tf"

    if os.path.exists("loss.csv"):
        losses = os.open("loss.csv", os.O_WRONLY | os.O_APPEND)
    else:
        losses = os.open("loss.csv", os.O_WRONLY | os.O_CREAT)
        os.write(losses, bytes("epoch,loss,validation loss, accuracy, validation accuracy\n", "UTF-8"))

    if os.path.exists(model_filename):
        model = tf.keras.models.load_model(model_filename)
    else:
        model = ImageGenerator().get_model()
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
                  loss=Huber(delta=0.5), metrics=['accuracy'])
    model.summary()

    #    validation_ids = list(enumerate(validation_dataset))
    #    random.shuffle(validation_ids)
    validation_ids = [42, 92, 117, 127, 155]

    expect, sources, validate = make_validation_data(validation_dataset, validation_ids)
    x = image_loader.load_image('/Users/riese/tmp/images/3343OO_screenshot.jpg', desired_size=(80, 80))
    y = image_loader.load_image('/Users/riese/tmp/images/3343OO.png', desired_size=(80, 80))
    x = np.array(x / 255, dtype=np.float32)
    y = np.array(y / 255, dtype=np.float32)
    validate = tf.concat([validate, np.reshape(normalize_x(deepcopy(x)), (1, 80, 80, 3))], axis=0)
    sources = np.concatenate([sources, x.reshape((1, 80, 80, 3))], axis=0)
    expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)

    display = ImageDisplay(with_graph=True)

    batch_size = 64
    epochs = config['epochs']

    print("loading traning samples, stand by...")
    if os.path.exists("inputs.pickle"):
        f = open("inputs.pickle", "rb")
        X, Y = pickle.load(f)
        f.close()
    else:
        X, Y = list(zip(*[dataset.__getitem__(i) for i in range(int(len(dataset) / 1))]))
        X = list([normalize_x(x) for x in X])
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        f = open("inputs.pickle", "wb")
        pickle.dump((X, Y), f)
        f.close()
    print(f"done loading {len(X)} samples.")

    val_inputs = val_expect = np.ndarray((0, 80, 80, 3))
    for i in range(len(validation_dataset)):
        x, y = validation_dataset[i]
        val_inputs = np.concatenate([val_inputs, x.reshape((1, 80, 80, 3))], axis=0)
        val_expect = np.concatenate([val_expect, y.reshape((1, 80, 80, 3))], axis=0)

    callback = PlottingCallback(display, sources, validate, expect, losses)

    model.fit(x=X, y=Y,
              steps_per_epoch=int(len(X) / batch_size),
              shuffle=True,
              epochs=epochs, validation_freq=1, verbose=1,
              validation_data=(val_inputs, val_expect),
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
