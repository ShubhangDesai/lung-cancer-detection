import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable

import numpy as np

from unet import unet

class UNet(object):
    def __init__(self, lr=1e-3, pos_weight=1):
        self.unet = unet(is_deconv=False)
        
        #weight = torch.from_numpy(np.array(weight)).float()
        #self.loss = nn.CrossEntropyLoss(weight=weight)
        self.pos_weight = pos_weight
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.unet.parameters(), lr=lr)
        
        if torch.cuda.is_available():
            self.loss.cuda()
            self.unet.cuda()
     
    def train(self, X, y):
        #l = self.loss(y_hat.permute(0, 2, 3, 1).contiguous().view(-1, 2), y.view(-1))
        
        self.unet.zero_grad()
        
        weight = torch.clamp(y*self.pos_weight, 1, self.pos_weight).data
        self.loss.weight = weight
        
        # Forward Pass
        y_hat = self.unet.forward(X)
        l = self.loss.forward(y_hat, y)
        
        # Backward Pass
        l.backward()
        self.optimizer.step()
        
        return l
    
    def evaluate(self, X, y):
        # Evaluaton
        y_hat = self.unet.forward(X)
        l = self.loss(y_hat, y)
        
        return l
    
    def pred(self, X):
        # Prediction
        y_hat = self.unet.forward(X)
        return y_hat
    
    def save(self, path='unet_ckpt'):
        torch.save(self.unet.state_dict(), "ckpts/" + path)
    
    def load(self, path):
        self.unet.load_state_dict(torch.load(path))