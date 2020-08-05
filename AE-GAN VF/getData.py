import numpy as np
import sys
import vtktools
#import pyvista as pv
from Variables import x_max, x_min
from Norm import normalise, denormalise

def get_velocity_field(fileNumber):
    """
    Used to get the Velocity Field as a numpy array corresponding to a vtu file from the Fluids dataset 
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :return: numpy array
        Tracers are returned as numpy array
    """
    folderPath = 'E:\MSc Individual Project\Fluids Dataset\small3DLSBU'
    #folderPath = '/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU'
    filePath = folderPath + '/LSBU_' + str(fileNumber) + '.vtu'
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)
    ug.GetFieldNames()
    p = ug.GetVectorField('Velocity')
    p = np.array(p)

    # Normalise p
    p = normalise(p, x_min, x_max)
    # Convert p into 1 x N array
    p = np.array(p)
    p = p.transpose()
    #p = np.array([p[:]])
    return p
p = get_velocity_field(500)
print(p.shape)

def get_tracer_from_latent(fileNumber):
    """
    Used to get the Tracers as a numpy array corresponding to a csv file from the Latent dataset 
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :return: numpy array
        Tracers are returned as numpy array
    """
    folderPath = 'E:/MSc Individual Project/Fluids Dataset/LatentSpace'
    #folderPath = '/vol/bitbucket/ja819/Fluids Dataset/LatentSpace'
    filePath = folderPath + '/LS_' + str(fileNumber) + '.csv'
    p = np.loadtxt(filePath, delimiter=",")

    return p

def get_prediction_tracer(fileNumber):
    """
    Used to get the Tracers as a numpy array corresponding to a vtu file from the Fluids dataset 
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :return: numpy array
        Tracers are returned as numpy array
    """
    folderPath = 'E:/MSc Individual Project/Fluids Dataset/predictionsGAN2'
    #folderPath = '/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU'
    filePath = folderPath + '/GAN_' + str(fileNumber) + '.vtu'
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)
    ug.GetFieldNames()
    p = ug.GetScalarField('PredictionGAN')
    p = np.array(p)

    # Normalise p
    p = normalise(p, x_min, x_max)
    # Convert p into 1 x N array
    p = np.array([p[:]])
    return p