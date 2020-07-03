import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from Variables import *

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn. ConvTranspose1d(4, 128, 4, 1, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Tanh()
        )