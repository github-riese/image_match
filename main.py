import sys

from bounding_box import detect_boxes

if __name__ == '__main__':
    if sys.argv[1] == 'box':
        detect_boxes(sys.argv[1:])
