import unittest
from PIL import Image
from PIL.ImageDraw import ImageDraw

from bounding_box.detector import detect


class DetectTest(unittest.TestCase):
    def test_that_simple_bounding_box_is_detected(self):
        image = Image.new("L", (200, 200), 255)
        draw = ImageDraw(image)
        draw.rectangle(((50, 50), (150, 150)), 128)
        box = detect(image)
        self.assertEqual(((.25, .25), (.75, .75)), box)

    def test_that_bounding_box_of_circle_is_detected(self):
        image = Image.new("L", (200, 200), 255)
        draw = ImageDraw(image)
        draw.arc(((50, 50), (100, 100)), 0, 360, 128)
        box = detect(image)
        self.assertEqual(((.25, .25), (.5, .5)), box)
