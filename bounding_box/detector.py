from typing import Tuple

import torch
import torchvision.transforms as T
import numpy as np
from torch.nn import Module


class Detector:
    _detector_model: Module
    _normalize: T.Normalize

    def __init__(self, model_path: str):
        self._detector_model = torch.load(model_path)
        self._detector_model.train(False)
        self._normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def detect_bounding_box(self, image: np.ndarray) -> Tuple[float, float, float, float]:
        image = np.transpose(image, (2, 0, 1))
        shape = image.shape
        image = image.reshape((1, shape[0], shape[1], shape[2]))
        if image.dtype in [np.uint8, np.int, int]:
            image = np.array(image, dtype=np.float32)
            image /= 255.
        tensor = torch.from_numpy(image)
        tensor = self._normalize(tensor)
        box = self._detector_model.forward(tensor)
        box = box[0]
        return box[0].item(), box[1].item(), box[2].item(), box[3].item()
