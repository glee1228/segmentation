import glob
from PIL import Image
import numpy as np
import os
import sys
from tqdm import tqdm
import shutil



if __name__ == '__main__':



    # arg_Model : which model to use
    # arg_DataRoot : path to the dataRoot
    # arg_thres : threshold of the image output from the model


    Thres = 200
    dataRoot = 'output{}'.format(Thres)
    saveRoot = 'total{}'.format(Thres)
    dataRoot = os.path.join(dataRoot)

    max_width = 3704
    max_height = 10000

    img_paths = glob.glob(os.path.join(dataRoot, '*.png'))

    img_paths = sorted(img_paths)
    img_lst = []
    print(img_paths)
    print('total number of patch Images : ',len(img_paths))
    if os.path.isdir(saveRoot):
        shutil.rmtree(saveRoot)
        os.mkdir(saveRoot)
    else:
        os.mkdir(saveRoot)
    roadnames = []
    for img_path in img_paths:
        fname = img_path.split('/')[-1]
        if len(fname.split('_')) == 7:
            roadname = '_'.join(fname.split('_')[:4])
            roadnames.append(roadname)

    roadnames=list(set(roadnames))
    print('total number of RoadNames : ',len(roadnames))

    for roadname in tqdm(roadnames):
        # print(img_paths)
        back = Image.new('RGB', (max_width, max_height), color='black')
        fullName = ''
        for idx, img_path in enumerate(img_paths):
            fname=img_path.split('/')[1]
            fname_lst=fname.split('_')
            if len(fname_lst)==7:
                road = '_'.join(fname_lst[:4])

                if roadname == road:
                    patch_size = int(fname_lst[4][1:])
                    x_start =  int(fname_lst[5])
                    y_start = int(fname_lst[6].split('.')[0])

                    fullName='{}'.format(roadname)
                    img=Image.open(img_path)
                    back.paste(img, (x_start, y_start))
        back.save('{}/{}.png'.format(saveRoot,fullName))
    print('All Full Road Images are written.')


