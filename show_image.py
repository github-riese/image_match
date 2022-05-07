from matplotlib import pyplot as plt
from matplotlib.image import AxesImage

__im: AxesImage = None


def show_image(image_array):
    global __im
    if __im is None:
        __im = plt.imshow(image_array)
    else:
        __im.set_array(image_array)
    plt.pause(.1)
