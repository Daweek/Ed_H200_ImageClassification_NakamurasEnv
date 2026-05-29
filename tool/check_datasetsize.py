import os
import argparse


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
    '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def main(args):
    for dataroot in args.dataroots:
        paths = []
        for root, dnames, fnames in os.walk(dataroot, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    paths.append(os.path.join(root, fname))
        print(f'{len(paths)} images in {dataroot}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroots', type=str, nargs='+', required=True)
    args = parser.parse_args()

    main(args)
