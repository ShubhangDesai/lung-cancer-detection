import os

import torch
from torch.utils.data.dataset import Dataset

import pandas as pd
import numpy as np

class ClassifierDataset(Dataset):
    def __init__(self, path):
        super(ClassifierDataset, self).__init__()
        self.path = path
        
        self.names = []
        for label in ['0', '1']:
              
            total_path = path + '/' + label
            for name in os.listdir(total_path):
                if '.npy' in name:
                    self.names.append((name.replace('.npy', ''), label))
    
    def __getitem__(self, index):
        name = self.names[index]
        #print(self.df[self.df['id'].str.contains(name)]['cancer'].as_matrix())
        #y = torch.from_numpy(self.df[self.df['id'].str.contains(name)]['cancer'].as_matrix())
        y = torch.from_numpy(np.array([int(name[1])]))
        
        
        X = np.load(self.path + '/' + str(y[0]) + '/' + name[0] + '.npy')
        X = X.reshape((1, 64, 64, 64))
        X = torch.from_numpy(X)
        
        return X, y
    
    def __len__(self):
        return len(self.names)