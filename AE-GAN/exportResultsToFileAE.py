##########################################################
# Fucntion to export results of a given test run to file #
##########################################################

import Variables
from AutoEncoder import Encoder, Decoder
from Generator import Generator
from mainAE import optimizerEnc, optimizerDec, loss_list, val_loss_list, epoch_list, val_ints

import pickle

def save_AE_vars(fileName, nameOfModel):
    pickle.dump([])




