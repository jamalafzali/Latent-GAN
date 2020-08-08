import numpy as np
import matplotlib.pyplot as plt
import csv

# Save graph outputs
filePath = '/vol/bitbucket/ja819/Python Files/Latent-GAN/AE-GAN/GANgraphs/GANUps.csv'

epoch_list = []
g_loss_list = []
d_loss_list = []
bce_loss_list = []
mse_loss_list = []
val_loss_list = []

csv_file = open(filePath, 'r')

for epoch, g, d, bce, mse, val in csv.reader(csv_file, delimiter=','):
    epoch_list.append(int(epoch))
    g_loss_list.append(float(g))
    d_loss_list.append(float(d))
    bce_loss_list.append(float(bce))
    mse_loss_list.append(float(mse))
    val_loss_list.append(float(val))

plt.plot(epoch_list, g_loss_list, label="Generator")
plt.plot(epoch_list, d_loss_list, label="Discriminator")
plt.plot(epoch_list, bce_loss_list, label="G_BCE Loss")
plt.plot(epoch_list, mse_loss_list, label="G_MSE Loss")
plt.plot(epoch_list, val_loss_list, label="Validation G_MSE")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()