# Functions to normalise and denormalise data
## Inputs can be numpy arrays or tensors

def normalise(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def denormalise(x_norm, x_min, x_max):
    return x_norm*(x_max - x_min) + x_min