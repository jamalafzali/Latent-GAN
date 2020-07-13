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

# Input will always be 1 X 100040
class AutoEncoder(nn.Module):
    def __init__(self, ngpu):
        super(AutoEncoder, self).__init__()
        self.ngpu = ngpu
        self.in_features = 100
        self.main = nn.Sequential(
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True), # Try LeakyReLU nn.LeakyReLU(0.2, inplace=True)
            nn.BatchNorm1d(ndf),
            # 
            nn.Conv1d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),
            nn.BatchNorm1d(ndf*2),

            nn.Conv1d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),
            nn.BatchNorm1d(ndf*4),

            nn.Conv1d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),
            nn.BatchNorm1d(ndf*8),

            nn.Conv1d(ndf*8, latent_size, 4, 2, 1, bias=False),
            #nn.MaxPool1d(4,2,1)
        )
    
    def forward(self, input):
        output = self.main(input)
        return output

#netAE = AutoEncoder(1).to("cuda:0")