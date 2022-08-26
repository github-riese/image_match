import math
import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from torch.nn import Module

from augmented_image_loader import AugmentedImageLoader
from generator.dataset import Dataset
from generator.generator_v2 import Generator, accuracy, NoisyNadam
from generator.tf_generator import ImageGenerator
from image_display import ImageDisplay
from image_loader import ImageLoader
from tf_dataloader_wrapper import normalize_x


class PlottingCallback(Callback):

    def __init__(self, display: ImageDisplay, validate: tf.Tensor, expect: np.ndarray,
                 loss_fd: int, model_filename: str):
        super(PlottingCallback, self).__init__()
        self._display = display
        self._validate = validate
        self._expect = expect
        self._loss_fd = loss_fd
        self._model_filename = model_filename

    def on_train_batch_end(self, batch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        bamm = self.model(tf.convert_to_tensor(self._validate, dtype=tf.float32))
        bamm = np.concatenate([self._validate.numpy(), bamm, self._expect], axis=0)
        self._display.show_images(bamm, int(math.ceil(bamm.shape[0] / 6)), losses=(logs['loss'], logs['val_loss']))
        self._display.save(f"snapshots/image_ep_{epoch + 1:03d}_{logs['loss']:.4f}.png")
        self.model.optimizer.epoch = epoch
        if epoch % 5 == 4:
            self.model.save(f"{self._model_filename}/theModel")
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
    image_loader = AugmentedImageLoader(image_loader=base_image_loader, size=(80, 80))

    data = pd.read_csv(config['csv_file'])
    dataset = Dataset(data, image_loader=image_loader)

    print(f"train dataset contains {len(dataset)} samples")

    model_filename = config['model']

    if os.path.exists("loss.csv"):
        losses = os.open("loss.csv", os.O_WRONLY | os.O_APPEND)
    else:
        losses = os.open("loss.csv", os.O_WRONLY | os.O_CREAT)
        os.write(losses, bytes("epoch,loss,validation loss, accuracy, validation accuracy\n", "UTF-8"))

    display = ImageDisplay(with_graph=True)

    epochs = config['epochs']

    print("loading traning samples, stand by...")
    if os.path.exists("inputs.pickle"):
        f = open("inputs.pickle", "rb")
        X, Y = pickle.load(f)
        f.close()
    else:
        X, Y = list(zip(*[dataset.__getitem__(i) for i in range(int(len(dataset) / 1))]))
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        f = open("inputs.pickle", "wb")
        pickle.dump((X, Y), f)
        f.close()
    print(f"done loading {len(X)} samples.")

    indices = [10, 1000, 20000, 17984, 666, 6660]
    validate = np.ndarray((0, 80, 80, 3), dtype=np.float32)
    expect = np.ndarray((0, 80, 80, 3), dtype=np.float32)
    for i in indices:
        x = X[i]
        y = Y[i]
        x = x.numpy()
        y = y.numpy()
        validate = np.concatenate([validate, x.reshape((1, 80, 80, 3))], axis=0)
        expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)
    for i in range(19, 60, 9):
        x = X[-i]
        y = Y[-i]
        x = x.numpy()
        y = y.numpy()
        validate = np.concatenate([validate, x.reshape((1, 80, 80, 3))], axis=0)
        expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)

    x = image_loader.load_image('/Users/riese/tmp/images/7024WC_screenshot.png', desired_size=(80, 80))
    y = image_loader.load_image('/Users/riese/tmp/images/7024WC.png', desired_size=(80, 80))
    x = np.array(x / 255, dtype=np.float32)
    y = np.array(y / 255, dtype=np.float32)
    validate = tf.concat([validate, np.reshape(x, (1, 80, 80, 3))], axis=0)
    expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)

    callback = PlottingCallback(display, validate, expect, losses, model_filename)

    epochs_done = config['initial_epoch']
    model = ensure_model(model_filename, latent_size=794)

    batch_size = 128
    beta_1 = .95
    noise_beta = .6
    lr_dampening = beta_1 ** epochs_done
    noise_dampening = noise_beta ** epochs_done
    learning_rate = 3.2e-4 * lr_dampening
    gradient_noise = 1e-6 * noise_dampening
    model.compile(optimizer=NoisyNadam(strength=gradient_noise, sustain=noise_beta,
                                       learning_rate=learning_rate,
                                       beta_1=beta_1, beta_2=0.8),
                  loss=model.loss,
                  metrics=[accuracy, 'mae'])
    model.fit(x=X, y=Y,
              steps_per_epoch=int(math.ceil(len(X) / batch_size / 1.9)),
              batch_size=batch_size,
              shuffle=True,
              epochs=epochs, validation_freq=1, verbose=1,
              validation_split=.1,
              validation_batch_size=batch_size,
              callbacks=callback, initial_epoch=epochs_done)  # config['initial_epoch'])

    model.save(f"{model_filename}/theModel_fixed", fixed=True)
    os.close(losses)
    plt.waitforbuttonpress()


def ensure_model(model_filename, latent_size: int = 256) -> tf.keras.Model:
    if os.path.exists(model_filename) and os.path.exists(model_filename + "/theModel.index"):
        model = Generator.load(filename=f"{model_filename}/theModel", latent_size=latent_size,
                               input_shape=(None, 80, 80, 3))
    else:
        if not os.path.exists(model_filename):
            os.mkdir(f"{model_filename}")
        model = Generator(latent_size=latent_size)
        model.build(input_shape=(None, 80, 80, 3))
    model.summary()
    return model


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
