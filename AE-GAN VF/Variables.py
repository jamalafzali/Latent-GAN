#######################################
# Setting Variables & Hyperparameters #
#######################################

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 4

# Number of channels in the training images.
nc = 3

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs_AE = 200
num_epochs_GAN = 100

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
x_max = 19.07114351017271
x_min = -11.642094691563011

# Validation Percentage (in decimals)
val_percent = 0.2

# Value to scale MSE Loss up during GAN training
alpha = 10