import torch
import torch.nn as nn

class TmpModel(nn.Module):
    def __init__(self):
        super(TmpModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,3,3,1,1),
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        return self.layer(x)