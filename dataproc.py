# import libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
from torchvision import transforms
# import torch modules
import torch
from torch.utils.data import Dataset
import random
from torchvision.utils import save_image
# BSDS dataset class for training data
import pdb
class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir, 
                 transform=None, target_transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.targetTransform = target_transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ')

    def __len__(self):
        return len(self.frame)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        # input and target images
        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])


        # process the images
        inputImage = Image.open(inputName).convert('RGB')

        # i, j, h, w = transforms.RandomCrop.get_params(inputImage, output_size=(256, 256))
        # inputImage = transforms.functional.crop(inputImage, i, j, h, w)
        if self.transform is not None:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed)
            inputImage = self.transform(inputImage)

        targetImage = Image.open(targetName).convert('L')
        # targetImage = transforms.functional.crop(targetImage, i, j, h, w)
        if self.targetTransform is not None:
            self._set_seed(seed)
            targetImage = self.targetTransform(targetImage)
        # save_image(inputImage, 'inputimg3/inp{}.jpg'.format(idx))
        # save_image(targetImage, 'inputimg3/inp{}.png'.format(idx))
        return inputImage, targetImage
    
# dataset class for test dataset
class TestDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None, target_transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.targetTransform = target_transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ',header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        fname = self.frame.iloc[idx, 0]
        inputName = os.path.join(self.rootDir, fname)
        targetName = None
        # process the images
        inputImage = Image.open(inputName).convert('RGB')
        # i, j, h, w = transforms.RandomCrop.get_params(inputImage, output_size=(256, 256))
        # inputImage = transforms.functional.crop(inputImage, i, j, h, w)


        if self.transform is not None:
            inputImage = self.transform(inputImage)

        if self.targetTransform is not None:
            targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])
            targetImage = Image.open(targetName).convert('L')
            # targetImage = transforms.functional.crop(targetImage, i, j, h, w)
            # targetImage_np = np.array(targetImage)
            # print(np.unique(targetImage_np,return_counts=True))
            targetImage = self.targetTransform(targetImage)
            # save_image(inputImage, 'inputimg/inp{}.jpg'.format(idx))
            # save_image(targetImage, 'inputimg/inp{}.png'.format(idx))
            return inputImage, fname , targetImage

        return inputImage, fname