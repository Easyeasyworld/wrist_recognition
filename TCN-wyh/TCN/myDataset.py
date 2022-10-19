import torch
from torch.utils.data import Dataset
import scipy.io as sio
import h5py
import numpy as np
from matplotlib import pyplot as plt

class myDataset(Dataset):

    def __init__(self, feattrain, feattest, labeltrain, labeltest, train=True):
        self.train = train
        self.feattrain = feattrain
        self.feattest = feattest
        self.labeltrain = labeltrain
        self.labeltest = labeltest

    def __getitem__(self, index):

        if self.train:
            return self.feattrain[index], self.labeltrain[index]-1
        else:
            return self.feattest[index], self.labeltest[index]-1

    def __len__(self):

        if self.train:
            return len(self.labeltrain)
        else:
            return len(self.labeltest)









