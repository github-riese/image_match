import os

import numpy as np
import pandas as pd
import torch
import torchinfo
from matplotlib import pyplot as plt
from torch.nn import Module, BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from augmented_image_loader import AugmentedImageLoader
from generator.dataset import Dataset
from generator.tf_generator import ImageGenerator
from image_display import ImageDisplay
from image_loader import ImageLoader
from tf_dataloader_wrapper import dataloader_wrapper
from trainer import Trainer

import tensorflow as tf


def generate_default_view(args: list):
    config = _configure(args)
    base_image_loader = ImageLoader({'jewellery': config['image_path']}, (240, 240))
    data = pd.read_csv(config['csv_file'])
    image_loader = AugmentedImageLoader(image_loader=base_image_loader, size=(80, 80))
    dataset = Dataset(data, reserve_percent=.925, image_loader=image_loader)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True)
    validation_dataset = dataset.get_reserved_data()
    validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False)

    if os.path.exists("models/tf_model.tf"):
        model = tf.keras.models.load_model("models/tf_model.tf")
    else:
        model = ImageGenerator().get_model()
        model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    v_iter = iter(validation_dataset)
    validate = tf.zeros((0, 80, 80, 3))
    expect = np.zeros((0, 80, 80, 3))

    for n in range(5):
        x, y = next(v_iter)
        validate = tf.concat([validate, x.reshape((1, 80, 80, 3))], axis=0)
        expect = np.concatenate([expect, y.reshape((1, 80, 80, 3))], axis=0)
    display = ImageDisplay()

    batch_size = 5
    for n in range(200):
        model.fit_generator(dataloader_wrapper(dataset, True, batch_size), len(dataset)/batch_size, epochs=5)
        bamm = model.predict(tf.convert_to_tensor(validate, dtype=tf.float32))
        bamm = np.concatenate([bamm, expect], axis=0)
        display.show_images(bamm, 5)
        model.save("models/tf_model.tf")
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
