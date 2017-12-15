import os

import torch
from torch.utils.data.dataset import Dataset

import numpy as np

class ScanDataset(Dataset):
    def __init__(self, img_path, gt_path):
        super().__init__()
        self.img_path = img_path
        self.gt_path = gt_path
        
        self.names = os.listdir(img_path)
    
    def __getitem__(self, index):
        name = self.names[index]

        X = torch.from_numpy(np.expand_dims(np.load(self.img_path + name), axis=0))
        y = torch.from_numpy(np.expand_dims(np.load(self.gt_path + name), axis=0))
        
        return X, y
    
    def __len__(self):
        return len(self.names)