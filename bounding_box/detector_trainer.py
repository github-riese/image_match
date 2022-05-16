import os.path
import time

import numpy as np
import pandas as pd
import torch
import torchinfo
import torchvision.transforms as T
from PIL import Image
from PIL.ImageDraw import ImageDraw
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary

from image_display import ImageDisplay
from image_loader import ImageLoader
from dataset import Dataset
from .detect_jewellery import DetectJewellery


def draw_boxes(X, Y, colour=(0, 255, 0)) -> np.ndarray:
    count, c, h, w = X.shape
    boxes = np.ndarray(shape=(0, h, w, c), dtype=np.uint8)
    for i in range(count):
        box = Y[i].detach().numpy()
        box = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
        box = min(box, (w, h, w, h))
        npimg = X[i].detach().numpy()
        npimg = np.transpose(npimg, (1, 2, 0)) * 255
        image = Image.fromarray(np.array(npimg, dtype=np.uint8))
        draw = ImageDraw(image)
        draw.rectangle(box, outline=colour)
        re_array = np.reshape(image, (1, h, w, c))
        boxes = np.append(boxes, re_array, axis=0)
    return boxes


def train_epoch(loader, model, loss_fn, optimizer, validation_loader, display):
    running_loss = 0
    started = time.time_ns()
    total_batches = len(loader)
    model.train(True)
    transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    print(f"stand by while training with {total_batches} batches...")
    epch_loss = 0
    for i, data in enumerate(loader):
        X, Y = data
        optimizer.zero_grad()
        predict = model.forward(transform(X))
        loss = loss_fn(predict, Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epch_loss += loss.item()
        if i % 5 == 4:
            now = time.time_ns()
            last_loss = running_loss / 5  # loss per batch
            running_loss = 0
            duration = (now - started) / (i * 1000000000)
            print(f"  batch {i + 1}/{total_batches} loss: {last_loss:.05f} - {duration:.02f} secs/batch")
            show(display, model, validation_loader, transform)
            torch.save(model, 'checkpoint.pt')
    return epch_loss / total_batches


def show(display, model, validation_loader, transform):
    model.train(False)
    for i, data in enumerate(validation_loader):
        X, Y = data
        predict = model.forward(transform(X))
        visual = draw_boxes(X, Y, colour=(90, 12, 77))
        visual = np.append(visual, draw_boxes(X, predict), axis=0)
        display.show_images(visual / 255., 10, grid_linewidth=2)
        break
    model.train(True)


def learn_boxes(args: list):
    config = configure(args)
    model = load_model()
    summary(model, input_size=(1, 3, 240, 240))  # input_data=torch.zeros((1, 3, 120, 120)))
    data = pd.read_csv(config['csv_in'])
    image_loader = ImageLoader({'jewellery': config['image_path']}, (240, 240))
    dataset = Dataset(data, reserve_percent=.925, image_loader=image_loader)
    validation_dataset = dataset.get_reserved_data()
    dataloader = DataLoader(dataset=dataset, batch_size=15, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=20, shuffle=False)
    log_fd = os.open("epoch_loss.csv", os.O_CREAT | os.O_WRONLY)
    os.write(log_fd, bytes("epoch,loss\n", "utf-8"))
    for epoch in range(50):
        print(f"Epoch {epoch + 1}...")
        loss = train_epoch(dataloader, model,
                           MSELoss(), Adam(model.parameters()),
                           validation_loader, ImageDisplay())
        print(f"Epoch {epoch + 1} - loss {loss:.05f}")
        os.write(log_fd, bytes(f"{epoch + 1},{loss:.05f}\n", "utf-8"))
    os.close(log_fd)


def configure(args: list) -> dict:
    return {
        'csv_in': args[1],
        'image_path': args[2]
    }


def load_model():
    file_name = os.path.realpath('.') + '/models/JewelleryDetect.pth'
    if os.path.exists(file_name):
        return torch.load(file_name)
    features = acquire_vgg16()
    detect = DetectJewellery(features)
    detect.init_weights()
    torch.save(detect, file_name)
    return detect


def acquire_vgg16():
    vgg16_original_filename = os.path.realpath('.') + '/models/VGG16-pretrained-original.pth'
    vgg16_no_classifier_filename = os.path.realpath('.') + '/models/VGG16-pretrained-no-classification.pth'
    if not os.path.exists(vgg16_original_filename):
        vgg = torch.hub.load('pytorch/vision:v0.12.0', 'vgg16', pretrained=True)
        torch.save(vgg, vgg16_original_filename)
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
