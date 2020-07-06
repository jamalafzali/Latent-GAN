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
            nn.Conv1d(1, int(tracer_input_size/16), 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

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