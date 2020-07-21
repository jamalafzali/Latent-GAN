#######################################
# Setting Variables & Hyperparameters #
#######################################

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 4

# Number of channels in the training images.
nc = 1

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 30

# Learning rate for optimizers
lr = 0.000007 # Using for MSE, works fine
#lr = 0.00001

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

# Defining minimum and maximums of dataset for normalistion
x_max = 1.0
x_min = -0.0011339544173616396

