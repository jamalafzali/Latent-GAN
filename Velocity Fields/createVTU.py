import sys
import vtktools
from Variables import *

def create_velocity_field_VTU_AE(fileNumber, prediction, networkName):
    """
    Duplicates preexisting VTU file and attaches a reconstruction field alongside it.
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :param prediction: numpy array
        Predicted tracers
    :param networkName: string
        Name of the network e.g. "AE", "GAN" or "AEGAN" etc.
        All created tracers will be saved to the default file directory in a
        folder with the network name.
    """
    folderPath = defaultFilePath + '/small3DLSBU'
    filePath = folderPath + '/LSBU_' + str(fileNumber) + '.vtu'
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)

    ug.AddVectorField('Latent-AE', prediction)

    saveFolderPath = defaultFilePath + '/' + networkName
    saveFolderPath = 'E:/MSc Individual Project/Fluids Dataset/tLatentGANExtrap'
    saveFilePath = saveFolderPath + '/' + networkName + '_' + str(fileNumber) + '.vtu'

    ug.Write(saveFilePath)

def create_velocity_field_VTU_GAN(fileNumber, prediction, networkName):
    """
    Duplicates preexisting VTU file and attaches a prediction field alongside it.
    Files are named based on the input timestep t.
    Files contain prediction for t+1 and ground truth for t+1.
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :param prediction: numpy array
        Predicted tracers
    :param networkName: string
        Name of the network e.g. "AE", "GAN" or "AEGAN" etc.
        All created tracers will be saved to the default file directory in a
        folder with the network name.
    """
    folderPath = defaultFilePath + '/small3DLSBU'
    filePath = folderPath + '/LSBU_' + str(fileNumber+1) + '.vtu'
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)

    ug.AddVectorField('Latent-GAN', prediction)

    saveFolderPath = defaultFilePath + '/' + networkName
    saveFolderPath = 'E:/MSc Individual Project/Fluids Dataset/tLatentGANExtrap'
    saveFilePath = saveFolderPath + '/' + networkName + '_' + str(fileNumber) + '.vtu'

    ug.Write(saveFilePath)
