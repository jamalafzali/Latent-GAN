import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from Variables import *

# Input will always be of size (1 x 100040)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nn.Conv1d(1, ndf * , 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Sigmoid()
        
            # input is (nc) x 64 x 64
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            
            nn.Conv1d(ndf * 8, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv1d(ndf * 32, ndf * 128, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(ndf * 128),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv1d(ndf * 128, 1, 4, 1, 0, bias=False),
            nn.Conv1d(ndf * 32, 1, 4, 1, 0, bias=False),
          
            nn.Sigmoid()

        )
    
    def forward(self, input):
        output = self.main(input)
        return output


# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.Conv1d(nc*4, 128, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Sigmoid()
#         )
    
#     def forward(self, input):
#         output = self.main(input)
#         return output