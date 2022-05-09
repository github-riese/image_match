import os.path

import pandas as pd
import torch
import torchinfo
from torchinfo import summary, model_statistics


def learn_boxes(args: list):
    config = configure(args)
    data = pd.read_csv(config['csv_in'])
    load_model()


def configure(args: list) -> dict:
    return {
        'csv_in': args[1]
    }


def load_model():
    vgg16_original_filename = 'VGG16-pretrained-original.pth'
    vgg16_no_classifier_filename = 'VGG16-pretrained-no-classification.pth'
    if not os.path.exists(vgg16_original_filename):
        vgg = torch.hub.load('pytorch/vision:v0.12.0', 'vgg16', pretrained=True)
        torch.save(vgg, 'VGG16-pretrained-original.pth')
    else:
        vgg = torch.load(vgg16_original_filename)

    if os.path.exists(vgg16_no_classifier_filename):
        sub = torch.load(vgg16_no_classifier_filename)
        return sub
    sub = torch.nn.Sequential()
    for name, net in vgg.named_children():
        if name == 'classifier':
            break
        sub.add_module(name, net)

    torch.save(sub, vgg16_no_classifier_filename)
    torchinfo.summary(sub, verbose=2)
    return sub
