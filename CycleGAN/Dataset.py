import torch
import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    '''
    ImageDataset class:
    This class is for receiving data and preparing them to give to the network.
        Values:
            root: Is the address or name of the dataset
            transfrom : It is a Transform function that will make changes to the image
    '''
    def __init__(self, root, transform = None, mode = "train"):
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode)+ '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        self.transform = transform
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you download the hazy2hazefree images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, item):
        item_A = self.transform(Image.open(self.files_A[item % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[item]]))
        if item_A.shape[0] !=3:
            item_A = item_A.repeat(3,1,1)
        if item_B.shape[0] !=3:
            item_B = item_B.repeat(3,1,1)
        if item == len(self) - 1:
            self.new_perm()
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))