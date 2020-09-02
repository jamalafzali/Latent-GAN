import numpy as np
import sys
import vtktools
#import pyvista as pv
from Variables import x_max, x_min, defaultFilePath
from Norm import normalise, denormalise
import pyvista as pv

def get_velocity_field(fileNumber):
    """
    Used to get the Velocity Field as a numpy array corresponding to a vtu file 
    from the Fluids dataset 
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :return: numpy array
        Velocity Fields are returned as numpy array
    """
    folderPath = defaultFilePath + '/small3DLSBU'
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

def get_velocity_field_structured(fileNumber):
    """
    Used to get the Velocity Field as a numpy array corresponding to a vtu file from the Fluids dataset 
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :return: numpy array
        Tracers are returned as numpy array
    """
    folderPath = defaultFilePath + '/small3DLSBU'
    filePath = folderPath + '/LSBU_' + str(fileNumber) + '.vtu'
    sys.path.append('fluidity-master')    mesh = pv.read(filePath)
    p = mesh.point_arrays['Velocity']

    # Normalise p
    p = normalise(p, x_min, x_max)
    # Convert p into 3 x N array
    p = np.array(p)
    p = p.transpose()
    #p = np.array([p[:]])
    return p

def get_prediction_velocity(fileNumber):
    """
    Used to get the Tracers as a numpy array corresponding to a vtu file from the Fluids dataset 
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :return: numpy array
        Tracers are returned as numpy array
    """
    networkName = 'prediction' # Change this to prediction folder name

    folderPath = defaultFilePath + '/' + networkName
    filePath = folderPath + '/' + networkName + '_' + str(fileNumber) + '.vtu' 
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)
    ug.GetFieldNames()
    p = ug.GetScalarField('PredictionGAN')
    p = np.array(p)

    # Normalise p
    p = normalise(p, x_min, x_max)
    # Convert p into 3 x N array
    p = np.array([p[:]])
    return p