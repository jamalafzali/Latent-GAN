import sys
import vtktools

def create_velocity_field_VTU(fileNumber, prediction, networkName):
    """
    Duplicates preexisting VTU file and attaches a prediction field alongside it
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :param prediction: numpy array
        Predicted tracers
    :param networkName: string
        Name of the network e.g. "AE", "GAN" or "AEGAN" etc.
    """
    folderPath = '/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU'
    #folderPath = 'E:/MSc Individual Project/Fluids Dataset/small3DLSBU'
    filePath = folderPath + '/LSBU_' + str(fileNumber+1) + '.vtu'
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)
    
    ug.AddVectorField('Prediction' + networkName, prediction)
    
    #saveFolderPath = '/vol/bitbucket/ja819/Fluids Dataset/predictions'
    saveFolderPath = 'E:/MSc Individual Project/Fluids Dataset/predictions'
    saveFilePath = saveFolderPath + networkName + '/' + networkName + '_' + str(fileNumber) + '.vtu'

    ug.Write(saveFilePath)
