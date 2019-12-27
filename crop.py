import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np


def make_dir(dir_name):
    try:
        if not (os.path.isdir(dir_name)):
            os.makedirs(os.path.join(dir_name))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

def without_hangul_filename(filename):
    without_hangul_list=filename.split('_')[1:]
    return "_".join(without_hangul_list)
def hangul_filepath_imread(filePath):
    stream = open(filePath.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='image file path, road name must be english',
                        default='C:\\Users\\user\\Downloads\\data_pothole_20191128\\images\\train')
    parser.add_argument('--output_dir', type=str, help='image output file',
                        default='C:\\Users\\user\\Downloads\\data_pothole_20191128\\croppedimg')
    parser.add_argument('--slice_size', type=int, help='input slice size',
                        default=256)
    parser.add_argument('--image_ext', type=str, help='set image extention',
                        default='jpg')
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = os.path.join(args.root_dir)
    file_list = os.listdir(image_path)
    image_list = [file for file in file_list if file.endswith("." + args.image_ext)]
    slice_size = args.slice_size
    shift = [[0,0]]
    make_dir(args.output_dir)

    # process image
    for start_x,start_y in shift:
        for image_name in image_list:
            image = hangul_filepath_imread(os.path.join(image_path, image_name))
            if len(image.shape)==3:
                height, width,channel = image.shape
            else :
                height, width = image.shape

            image_name_without_ext = image_name.split('.')[0]
            if args.image_ext=='jpg':
                image_name_without_ext = image_name.split('.')[0]
            elif args.image_ext=='png':
                image_name_without_ext = image_name.split('.')[0]

            make_dir(args.output_dir)
            for y in range(start_y, height - slice_size, slice_size):
                for x in range(start_x, width - slice_size, slice_size):
                    crop = image[y:y + slice_size, x:x + slice_size]
                    image_name_without_hangul_ext=without_hangul_filename(image_name_without_ext)
                    out_path = os.path.join(args.output_dir,image_name_without_hangul_ext + '_' + str(x) + '_' + str(y) + "." + args.image_ext)
                    if args.image_ext == 'jpg':
                        cv2.imwrite(out_path, crop)
                        print('asdf')
                    if args.image_ext == 'png':
                        if len(np.unique(crop))!=1:
                            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            crop[crop == 255] = 0
                            crop[crop==25]=0
                            crop[crop == 51] = 0
                            crop[crop==127]=0
                            crop[crop == 102] = 255
                            cv2.imwrite(out_path, crop)
                            # import pdb;pdb.set_trace()
                            print(out_path)


            print('image ', image_name, ' is sliced')


if __name__ == '__main__':
    main()