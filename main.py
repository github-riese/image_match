import sys
from bounding_box import find_bounding_boxes, learn_boxes
from generator import generate_default_view


def handle_boxes(args: list):
    if args[1] == 'find':
        find_bounding_boxes(args[1:])
    elif args[1] == 'learn':
        learn_boxes(args[1:])


def handle_generate(args: list):
    generate_default_view(args)


if __name__ == '__main__':
    if sys.argv[1] == 'box':
        handle_boxes(sys.argv[1:])
    if sys.argv[1] == 'generate':
        handle_generate(sys.argv[1:])
