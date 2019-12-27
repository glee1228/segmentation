# Author: sogang-mm
# Date: 2019/12/13

# import torch libraries
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# import utility functions
from model import *
from trainer import Trainer
from dataproc import TrainDataset
from util import make_txt
import random

# fix random seed
rng = np.random.RandomState(37148)

# GPU ID
gpuID = 0

# batch size
nBatch = 32

# max epoch
nEpoch = 250

# load the images dataset
dataRoot = 'data/crack/'
modelPath = 'model/vgg16.pth'

valPath = dataRoot+'val.lst'
trainPath = dataRoot+'train.lst'

# write txt file
make_txt(dataRoot,'train')
make_txt(dataRoot,'val')
make_txt(dataRoot,'test')

# create data loaders from dataset
std=[0.229, 0.224, 0.225]
mean=[0.485, 0.456, 0.406]

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
targetTransform = transforms.Compose([
                transforms.ToTensor()
            ])
#
# trans = transforms.Compose([
#                 transforms.RandomChoice([
#                     transforms.RandomRotation((0, 0)),
#                     transforms.RandomHorizontalFlip(p=1),
#                     transforms.RandomVerticalFlip(p=1),
#                     transforms.RandomRotation((90, 90)),
#                     transforms.RandomRotation((180, 180)),
#                     transforms.RandomRotation((270, 270)),
#                     transforms.Compose([
#                         transforms.RandomHorizontalFlip(p=1),
#                         transforms.RandomRotation((90, 90)),
#                     ]),
#                     transforms.Compose([
#                         transforms.RandomHorizontalFlip(p=1),
#                         transforms.RandomRotation((270, 270)),
#                     ])
#                 ])])

# transform = transforms.Compose([
#                 trans,
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean,std)
#             ])
# targetTransform = transforms.Compose([
#                 trans,
#                 transforms.ToTensor()
#             ])
transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
targetTransform_val = transforms.Compose([
                transforms.ToTensor()
            ])
valDataset = TrainDataset(valPath, dataRoot,
                          transform_val, targetTransform_val)
trainDataset = TrainDataset(trainPath, dataRoot,
                            transform, targetTransform)

valDataloader = DataLoader(valDataset, shuffle=False)
trainDataloader = DataLoader(trainDataset, shuffle=True)

# initialize the network
net = HED()
net.apply(weights_init)

pretrained_dict = torch.load(modelPath)
pretrained_dict = convert_vgg(pretrained_dict)

model_dict = net.state_dict()
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

net.cuda(gpuID)

# define the optimizer
lr = 1e-5
lrDecay = 0.1
lrDecayEpoch = list(range(1,999,1))

fuse_params = list(map(id, net.fuse.parameters()))
conv5_params = list(map(id, net.conv5.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params+fuse_params,
                     net.parameters())

# optimizer = torch.optim.SGD([
#             {'params': base_params},
#             {'params': net.conv5.parameters(), 'lr': lr * 10},
#             {'params': net.fuse.parameters(), 'lr': lr * 0.001}
#             ], lr=lr,momentum=0.9)

optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': net.conv5.parameters(), 'lr': lr * 10},
            {'params': net.fuse.parameters(), 'lr': lr * 0.001}
            ], lr=lr)

# initialize trainer class
trainer = Trainer(net, optimizer, trainDataloader, valDataloader, out='train',
                  nBatch=nBatch, maxEpochs=nEpoch, cuda=True, gpuID=gpuID,
                  lrDecay=lrDecay,lrDecayEpochs=lrDecayEpoch)

# train the network
trainer.train()
