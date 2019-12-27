import glob
from PIL import Image
import numpy as np
import os
import sys
dataRoot = 'data/MUHAN_512'
dataRoot = os.path.join(dataRoot)

imgRoot = os.path.join(dataRoot,'croppedimg')
gtRoot = os.path.join(dataRoot,'croppedgt')
outputRoot = os.path.join('output')

img_paths = glob.glob(os.path.join(imgRoot,'*.jpg'))
img_paths = sorted(img_paths)
gt_paths = glob.glob(os.path.join(gtRoot,'*.png'))
gt_paths = sorted(gt_paths)

img_lst = []
gt_lst = []
print(img_paths)
print(gt_paths)

for idx,img_path in enumerate(img_paths):
    gt_path=img_path.replace('croppedimg','croppedgt').replace('jpg','png')
    img_name = img_path.split('/')[-1][:-4]

    try:
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path)
        gt_np = np.array(gt)
        print('------------------------------------------------------------------------------------------')
        print('{} Before Data Preprocessing : {}'.format(img_name, np.unique(gt_np, return_counts=True)))

        gt_np[gt_np == 255] = 100
        gt_np[gt_np != 100] = 255
        gt_np[gt_np == 100] = 0

        print(np.unique(gt_np))

        print('{} After Data Preprocessing : {}'.format(img_name, np.unique(gt_np, return_counts=True)))
        print('-------------------------------------------------------------------------------------------')
        gt = Image.fromarray(gt_np.astype('uint8'))
        gt.putalpha(255)
        gt.save(os.path.join(outputRoot, '{}.png'.format(img_name)))
    except OSError as err:
        os.remove(img_path)
        os.remove(gt_path)
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not convert data to an integer.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise


