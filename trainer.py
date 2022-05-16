import time

import torch
import torchvision.transforms as T
from torch.nn import Module
from torch.utils.data import DataLoader

from image_display import ImageDisplay


class Trainer:
    def __init__(self, model: Module, loss_fn, optimizer):
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._im_display = ImageDisplay()
        self._normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def fit(self, train_set: DataLoader, num_epochs: int,
            validation_loader: DataLoader, verbose: bool):
        total_loss = 0
        for epoch in range(num_epochs):
            if verbose:
                print(f"epoch {epoch + 1}")
            self._model.train(True)
            epoch_loss = self._train_epoch(train_set, verbose, validation_loader)
            total_loss += epoch_loss
            if verbose and validation_loader is not None:
                self._show_validation(validation_loader)

    def _train_epoch(self, train_data: DataLoader, verbose: bool, validate):
        epoch_loss = 0
        self._optimizer.zero_grad()
        num_batches = len(train_data)
        started = time.time()
        for i, data in enumerate(train_data):
            X, Y = data
            predict = self._model.forward(self._normalize(X))
            batch_loss = self._loss_fn(predict, Y)
            batch_loss.backward()
            epoch_loss += batch_loss
            if verbose and i % 5 == 4:
                if verbose:
                    print(f" {i + 1}/{num_batches}: loss {epoch_loss / (i + 1):.5f}, "
                          f"{(time.time() - started) / (i + 1):.1f} s per batch.")
                torch.save(self._model, "checkpoint.pt")
                self._show_validation(validate)

        return epoch_loss / num_batches

    def _show_validation(self, validation_set):
        data = next(iter(validation_set))
        X, Y = data
        self._model.train(False)
        actual = self._model.forward(self._normalize(X))
        expected = torch.cat((X, actual, Y), dim=0)
        self._im_display.show_images(expected, X.shape[0])
