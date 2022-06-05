import re
from os import *
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw

from bounding_box.scan_dir import scan_dir


def draw_bounding_box(image: Image.Image,
                      box: Tuple[Tuple[float, float], Tuple[float, float]],
                      colour: Tuple[int, int, int, int] = (0, 255, 0, 0), ):
    draw = ImageDraw(image)
    xy = np.asarray(box)
    xy *= image.size

    draw.rectangle(((xy[0, 0], xy[0, 1]), (xy[1, 0], xy[1, 1])), outline=colour)


def find_bounding_boxes(args: List[str]) -> None:
    config = parse_args(args)
    print(config)
    csv = open(config['csv_file'], O_CREAT | O_WRONLY)
    write(csv, b"product_id,path,type,flavour,angle,x1,y1,x2,y2\n")
    expr = re.compile(r"(?P<product>\d{4}[A-Z]{2})_?(?P<angle>-?\d{1,3})?.png$")
    for f in scan_dir(config['img_path'], need_read_path):
        relative_path = f.replace(config['img_path'], '')
        if relative_path[0] == '/':
            relative_path = relative_path[1:]
        matches = expr.findall(relative_path)
        if len(matches) < 1:
            print(f"{relative_path} seem to be none of my images.")
            continue
        parts = relative_path.split('/')
        if len(parts) < 2:
            continue
        flavour, jewellery_type = parts[0], parts[1]

        img = read_image(f)
        box = find_bounding_box_in_image(img, l_threshold=220)
        if box == ((0.0, 0.0), (0.0, 0.0)):
            continue

        matches = expr.findall(f)
        product_id = matches[0][expr.groupindex['product'] - 1]
        angle = matches[0][expr.groupindex['angle'] - 1]
        if angle == '':
            angle = '0'
        write(csv,
              bytearray(
                  f"{product_id},{relative_path},{jewellery_type},{flavour},{angle},"
                  f"{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]}\n",
                  "UTF-8"))


def read_image(file_path: str) -> Image.Image:
    return Image.open(file_path)


def find_bounding_box_in_image(image: Image.Image, l_threshold: int = 246) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    reduced = image.convert("L")
    pixels = np.asarray(reduced)
    shape_pixels = np.transpose(np.nonzero(pixels < l_threshold))
    if len(shape_pixels) == 0:
        return (0.0, 0.0), (0.0, 0.0)
    shape_min = shape_pixels.min(0)
    shape_max = shape_pixels.max(0)
    box = np.asarray([
        [shape_min[1], shape_min[0]],
        [shape_max[1], shape_max[0]]
    ], dtype=float) / (image.width, image.height)
    return (box[0, 0], box[0, 1]), (box[1, 0], box[1, 1])


def parse_args(args: List[str]) -> Dict[str, str]:
    return {
        'img_path': args[1],
        'csv_file': args[2]
    }


def need_read_path(directory: DirEntry, depth: int) -> bool:
    if depth != 0:
        return True
    return directory.name in ['original', 'frame', 'transformed']
