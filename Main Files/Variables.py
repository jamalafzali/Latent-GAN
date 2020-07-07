#####################
# Setting Variables #
#####################

# Root directory for dataset
dataroot = ".\celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 8

# Spatial size of training images. All images will be resized to this size using a transformer
image_size = 64

# Number of channels in the training images.
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU.
ngpu = 1

# Size of tracer input data
tracer_input_size = 100040

# Number of time steps
time_steps = 988

# Latent Space Size
latent_size = 128 # Use 1024? ~ 100040 / 2*6