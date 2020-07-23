import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from Variables import *
from getData import get_tracer, get_tracer_from_latent

########################
# Creating the dataset #
########################
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class TracerDataset(Dataset):
    """Tracer Dataset"""

    def __init__(self, file_number='', root_dir='', transform=None):
        """
        Initialise the Dataset
        :fileNumber: int or string
            Used to specify which file to open
        : rootDir: string
            Directory of all vtu files
        :transform: callable, optional
            Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = time_steps # number of timesteps available

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = get_tracer(idx)
        #sample = get_tracer_from_latent(idx)
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        # swap axis because
        # numpy: H x W x C
        # torch: C x H x W
        
        #print("Size is ", sample.shape())
        #sample = sample.transpose((2, 0, 1)) # May need to flip this for Veloctiy
        sample = torch.from_numpy(sample)
        return sample

class TracerLatentDataset(Dataset):
    """Tracer Dataset"""

    def __init__(self, file_number='', root_dir='', transform=None):
        """
        Initialise the Dataset
        :fileNumber: int or string
            Used to specify which file to open
        : rootDir: string
            Directory of all vtu files
        :transform: callable, optional
            Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = time_steps # number of timesteps available

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #sample = get_tracer(idx)
        sample = get_tracer_from_latent(idx)
        if self.transform:
            sample = self.transform(sample)

        return sample

