import numpy as np
import os
import glob
import random


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''
    print(type(eval_segm),type(gt_segm))
    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_val_ = np.sum(IU) / n_cl_gt
    return mean_IU_val_


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)
    print(h_e,w_e)
    print(h_g, w_g)
    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def randomSample(img_paths,gt_paths,numOfSample):
    sample_img_paths = []
    sample_gt_paths = []
    sampling = random.sample(img_paths, numOfSample)
    for img_path, gt_path in zip(img_paths, gt_paths):
        if img_path in sampling:
            img_paths.remove(img_path)
            gt_paths.remove(gt_path)
            sample_img_paths.append(img_path)
            sample_gt_paths.append(gt_path)
    print(len(img_paths))
    return img_paths,gt_paths,sample_img_paths,sample_gt_paths


def write_txt(dataRoot,filename,img_paths,gt_paths):
    print(img_paths)
    txtfile = open(os.path.join(dataRoot,filename),'w')
    for img_path, gt_path in zip(img_paths,gt_paths):
        saved_img = os.path.relpath(img_path, dataRoot)
        saved_gt = os.path.relpath(gt_path, dataRoot)
        txtfile.write('{} {}\n'.format(saved_img, saved_gt))

    txtfile.close()

def make_txt(dataRoot,dataType):
    img_paths = glob.glob(os.path.join(dataRoot, dataType,'croppedimg/*.jpg'))
    gt_paths = glob.glob(os.path.join(dataRoot, dataType, 'croppedgt/*.png'))
    img_paths = sorted(img_paths)
    gt_paths = sorted(gt_paths)

    write_txt(dataRoot,dataType+'.lst',img_paths,gt_paths)


def random_make_txt(dataRoot,numOfVal,numOfTest):
    img_paths = glob.glob(os.path.join(dataRoot, 'croppedimg/*.jpg'))
    gt_paths = glob.glob(os.path.join(dataRoot, 'croppedgt/*.png'))
    img_paths = sorted(img_paths)
    gt_paths = sorted(gt_paths)

    img_paths, gt_paths, val_img_paths, val_gt_paths = randomSample(img_paths,gt_paths,numOfVal)
    img_paths, gt_paths, test_img_paths, test_gt_paths = randomSample(img_paths,gt_paths,numOfTest)

    write_txt(dataRoot,'train_pair.lst',img_paths,gt_paths)
    write_txt(dataRoot, 'val_pair.lst', val_img_paths, val_gt_paths)
    write_txt(dataRoot, 'test.lst', test_img_paths, test_gt_paths)
    print('train, validation, test is written')

def writeAll(dataRoot):
    print(dataRoot)
    txtfile = open(os.path.join(dataRoot, 'test.lst'), 'w')
    img_paths = glob.glob(os.path.join(dataRoot, 'test/croppedimg/*.jpg'))

    img_paths = sorted(img_paths)

    for img_path in img_paths:
        saved_img = os.path.relpath(img_path, dataRoot)
        saved_gt = saved_img.replace('croppedimg', 'croppedgt').replace('jpg', 'png')
        txtfile.write('{} {}\n'.format(saved_img, saved_gt))

    txtfile.close()
    print('All dataset list is written')



