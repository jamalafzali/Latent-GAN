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
        
            # input is 1 x 100040
            nn.Conv1d(nc, ndf, 4, 8, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ndf x 12,505

            nn.Conv1d(ndf, ndf * 2, 4, 8, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 1563
            
            nn.Conv1d(ndf * 2, ndf * 4, 4, 8, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 195
            
            nn.Conv1d(ndf * 4, ndf * 8, 4, 8, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 24
            
            nn.Conv1d(ndf * 8, ndf * 16, 4, 4, 1, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 6

            nn.Conv1d(ndf * 16, ndf * 32, 4, 4, 1, bias=False),
            nn.BatchNorm1d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 1

            nn.Conv1d(ndf * 32, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()

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