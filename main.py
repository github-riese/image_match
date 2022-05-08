import sys

from bounding_box import find_bounding_boxes, learn_boxes


def handle_boxes(args: list):
    if args[1] == 'find':
        find_bounding_boxes(args[1:])
    elif args[1] == 'learn':
        learn_boxes(args[1:])


if __name__ == '__main__':
    if sys.argv[1] == 'box':
        handle_boxes(sys.argv[1:])
