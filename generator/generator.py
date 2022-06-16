import math
import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.callbacks import Callback
from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
from torch.nn import Module

from augmented_image_loader import AugmentedImageLoader
from generator.dataset import Dataset
from generator.generator_v2 import Generator
from generator.tf_generator import ImageGenerator
from image_display import ImageDisplay
from image_loader import ImageLoader
from tf_dataloader_wrapper import normalize_x


class PlottingCallback(Callback):

    def __init__(self, display: ImageDisplay, validate: tf.Tensor, expect: np.ndarray,
                 loss_fd: int):
        super(PlottingCallback, self).__init__()
        self._display = display
        self._validate = validate
        self._expect = expect
        self._loss_fd = loss_fd

    def on_train_batch_end(self, batch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        bamm = self.model.predict(tf.convert_to_tensor(self._validate, dtype=tf.float32))
        bamm = np.concatenate([self._validate.numpy(), bamm, self._expect], axis=0)
        self._display.show_images(bamm, int(math.ceil(bamm.shape[0] / 3)), losses=(logs['loss'], logs['val_loss']))
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
    image_loader = AugmentedImageLoader(image_loader=base_image_loader, size=(80, 80))

    data = pd.read_csv(config['csv_file'])
    dataset = Dataset(data, image_loader=image_loader)

    print(f"train dataset contains {len(dataset)} samples")

    model_filename = config['model']
    batch_size = 32

    if os.path.exists("loss.csv"):
        losses = os.open("loss.csv", os.O_WRONLY | os.O_APPEND)
    else:
        losses = os.open("loss.csv", os.O_WRONLY | os.O_CREAT)
        os.write(losses, bytes("epoch,loss,validation loss, accuracy, validation accuracy\n", "UTF-8"))

    if os.path.exists(model_filename):
        model = Generator.load(filename=model_filename)
#        model = tf.keras.models.load_model(model_filename, compile=False,
#                                           custom_objects={'CustomModel': 'generator.generator_v2.Generator'})
    else:
        model = Generator(latent_size=256)
    model.build(input_shape=(None, 80, 80, 3))
    model.compile(optimizer=Adam(learning_rate=2.5e-6, beta_1=0.5, beta_2=0.75), loss=model.loss,
                  metrics=['accuracy', 'mse'])
    model.summary()

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

    validate = np.ndarray((0, 80, 80, 3), dtype=np.float32)
    expect = np.ndarray((0, 80, 80, 3), dtype=np.float32)
    for i in range(5):
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

    callback = PlottingCallback(display, validate, expect, losses)

    model.fit(x=X, y=Y,
              steps_per_epoch=int(math.ceil(len(X) / batch_size / 3)),
              shuffle=True,
              epochs=epochs, validation_freq=1, verbose=1,
              validation_split=.1,
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
