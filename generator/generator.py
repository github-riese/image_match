import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.optimizer_v2.nadam import Nadam
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import History
from torch.nn import Module
from torch.utils.data import DataLoader

from augmented_image_loader import AugmentedImageLoader
from generator.dataset import Dataset
from generator.tf_generator import ImageGenerator
from image_display import ImageDisplay
from image_loader import ImageLoader
from tf_dataloader_wrapper import dataloader_wrapper


def generate_default_view(args: list):
    config = _configure(args)
    base_image_loader = ImageLoader({'jewellery': config['image_path']}, (240, 240))
    data = pd.read_csv(config['csv_file'])
    image_loader = AugmentedImageLoader(image_loader=base_image_loader, size=(80, 80))
    dataset = Dataset(data, reserve_percent=.925, image_loader=image_loader)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True)
    validation_dataset = dataset.get_reserved_data()
    validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False)

    if os.path.exists("models/tf_model_conv.tf"):
        model = tf.keras.models.load_model("models/tf_model_conv.tf")
    else:
        model = ImageGenerator().get_model()
    model.compile(optimizer=Nadam(0.01), loss='binary_crossentropy')
    model.summary()

    #    validation_ids = list(enumerate(validation_dataset))
    #    random.shuffle(validation_ids)
    validation_ids = [5, 15, 25, 35, 45]

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

    batch_size = 15
    total_epochs = 0
    for n in range(20):
        print(f"meta-epoch {n + 1}...")
        epochs = min(n * 2 + 1, 10)
        history: History = model.fit(dataloader_wrapper(dataset, True, batch_size),
                                     steps_per_epoch=len(dataset) / batch_size,
                                     epochs=epochs, validation_freq=1, verbose=1)
        bamm = model.predict(tf.convert_to_tensor(validate, dtype=tf.float32))
        bamm = np.concatenate([sources, bamm, expect], axis=0)
        display.show_images(bamm, 5)
        display.save(f"snapshots/image_ep_{n + 1}_{history.history['loss'][-1]:.4f}.png")
        model.save("models/tf_model_conv.tf")
        print(f"meta-epoch {n + 1}: avg_loss: {sum(history.history['loss'])/len(history.history['loss']):.4f}")
        total_epochs += epochs
    print(f"done 20 meta-epochs with {total_epochs} epochs run.")
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
