#import torch

def normalise(x, x_min, x_max):
    #x = x.numpy()
    return (x - x_min) / (x_max - x_min)

def denormalise(x_norm, x_min, x_max):
    #x_norm = x_norm.numpy()
    return x_norm*(x_max - x_min) + x_min
    