from math import ceil
from typing import Tuple

import numpy
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from torch import Tensor


class ImageDisplay:
    __im: AxesImage = None

    def __init__(self):
        plt.axis('off')

    def show_image(self, image: np.ndarray):
        if self.__im is None:
            self.__im = plt.imshow(image)
        else:
            self.__im.set_array(image)
        plt.pause(.1)

    def show_images(self, images: np.ndarray, columns: int = 1, grid_linewidth: int = 1):
        if isinstance(images, Tensor):
            images = images.detach().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
        if images.dtype != np.uint8:
            images = np.array(images * 255, dtype=np.uint8)
        count, height, width, channels = images.shape
        rows = int(ceil(count / columns))
        grid = self._make_grid(images, columns, rows, width, height, line_width=grid_linewidth)
        self.show_image(grid)

    @staticmethod
    def _make_grid(images: np.ndarray, columns: int, rows: int, im_width: int, im_height: int,
                   padding: int = 5, line_width: int = 1) -> np.ndarray:
        w = columns * (im_width + padding)
        h = rows * (im_height + padding)
        grid = Image.new(mode='RGB', size=(w, h), color=(255, 255, 255))
        draw = ImageDraw(grid)
        stride_x, stride_y = im_width + padding, im_height + padding
        offset = 0
        for y in range(rows):
            for x in range(columns):
                image = images[offset] if offset < len(images) else None
                offset += 1
                top = y * stride_y
                left = x * stride_x
                right, bottom = (x + 1) * stride_x, (y + 1) * stride_y
                if image is not None:
                    paste = Image.fromarray(image)
                    p_left, p_top = int(left + padding / 2) + 1, int(top + padding / 2) + 1
                    p_right, p_bottom = p_left + image.shape[1], p_top + image.shape[0]
                    grid.paste(paste, (p_left, p_top, p_right, p_bottom))
                rect = ImageDisplay._adjust_grid((h, w), (left, top, right, bottom), line_width)
                draw.rectangle(rect,
                               outline=(0, 0, 0), width=line_width)
        return grid.__array__(dtype=float) / 255

    @staticmethod
    def _adjust_grid(canvas_size: Tuple[int, int],
                     box: Tuple[int, int, int, int],
                     line_width: int) -> Tuple[int, int, int, int]:
        left, top, right, bottom = box
        h, w = canvas_size
        if left <= 0:
            left = line_width
        if left >= w:
            left = w - line_width
        if right <= 0:
            right = line_width
        if right >= w:
            right = w - line_width
        if top <= 0:
            top = line_width
        if top >= h:
            top = h - line_width
        if bottom <= 0:
            bottom = line_width
        if bottom >= h:
            bottom = h - line_width
        return left, top, right, bottom
