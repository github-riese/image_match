from matplotlib import pyplot as plt
from matplotlib.image import AxesImage


class ImageDisplay:
    __im: AxesImage = None

    def show_image(self, image_array):
        if self.__im is None:
            self.__im = plt.imshow(image_array)
        else:
            self.__im.set_array(image_array)
        plt.pause(.1)
