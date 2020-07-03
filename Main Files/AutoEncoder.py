import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from Variables import *

# class AutoEncoder(nn.Module):
#     def __init__(self, ngpu):
#         super(AutoEncoder, self).__init__()
#         self.ngpu = ngpu
#         self.in_features = 100040
#         self.main = nn.Sequential(
#             nn.Conv2d(int(self.in_features), 
#                         int(self.in_features/2), kernel_size=4, 
#                         stride=2, padding=1, bias = False)
#         )
    
#     def forward(self, input):
#         output = self.main(input)
#         return output

class AutoEncoder(nn.Module):
    def __init__(self, ngpu):
        super(AutoEncoder, self).__init__()
        self.ngpu = ngpu
        self.in_features = 100
        self.main = nn.Sequential(
            nn.Conv1d(1, 
                        4, kernel_size=4, 
                        stride=2, padding=1, bias = False)
        )
    
    def forward(self, input):
        output = self.main(input)
        return output

netAE = AutoEncoder(1).to("cuda:0")