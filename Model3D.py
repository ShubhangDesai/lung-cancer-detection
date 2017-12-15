import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Variable

import numpy as np

class Model3D(object):
    def __init__(self, model, lr=1e-3, pos_weight=1):
        self.model = model
        
        self.pos_weight = pos_weight
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        
        if torch.cuda.is_available():
            self.loss.cuda()
            self.model.cuda()
            
    def train(self, X, y):
        self.model.zero_grad()
        
        weight = torch.clamp(y*self.pos_weight, 1, self.pos_weight).data
        self.loss.weight = weight
        
        # Forward Pass
        y_hat = self.model.forward(X)
        l = self.loss.forward(y_hat, y)
        
        # Backward Pass
        l.backward()
        self.optimizer.step()
        
        return l
    
    def evaluate(self, X, y):
        # Evaluaton
        y_hat = self.model.forward(X)
        l = self.loss(y_hat, y)
        
        return l
    
    def pred(self, X):
        # Prediction
        y_hat = self.model.forward(X)
        return y_hat
    
    def save(self, path='3d_model_ckpt'):
        torch.save(self.model.state_dict(), "ckpts/" + path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))