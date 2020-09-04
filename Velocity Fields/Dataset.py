import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from Variables import *
from getData import get_velocity_field, get_velocity_field_structured
from convertToStructuredMesh import get_structured_velocity

########################
# Creating the dataset #
########################
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class VelocityFieldDataset(Dataset):
    """Velocity Field Dataset"""

    def __init__(self, file_number='', transform=None):
        """
        Initialise the Dataset
        :fileNumber: int or string
            Used to specify which file to open
        :transform: callable, optional
            Optional transform to be applied on a sample
        """
        self.transform = transform
        self.length = time_steps # number of timesteps available

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = get_velocity_field(idx)
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        sample = torch.from_numpy(sample)
        return sample

class VelocityFieldDatasetStructured(Dataset):
    """Velocity Field Structured Dataset"""

    def __init__(self, file_number='', transform=None):
        """
        Initialise the Dataset
        :fileNumber: int or string
            Used to specify which file to open
        :transform: callable, optional
            Optional transform to be applied on a sample
        """
        self.transform = transform
        self.length = time_steps # number of timesteps available

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = get_structured_velocity(idx)
        if self.transform:
            sample = self.transform(sample)

        return sample