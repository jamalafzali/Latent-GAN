#from datafile import output
import sys
import vtktools

#contador = 0
#handicap = 60

#for i in range(1501):
# directory_model = '/vol/bitbucket/ja819/Python Files/Latent-GAN/AE-GAN/'

# # folderPath = '/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU'
# # filePath = folderPath + '/LSBU_' + str(fileNumber) + '.vtu'

# ug = vtktools.vtu('/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU/LSBU_83.vtu')
# #ug = vtktools.vtu(directory_model + 'Velocity2d_' + str(i)+'.vtu')

# #ug.AddScalarField('PredictionAE', rom[i, :])
# ug.AddScalarField('PredictionAE', output)

# #ug.RemoveField('ROM')
#  #ug.AddField('ROM', rom[contador + handicap, :])
# ug.Write(directory_model + 'test.vtu')

# #contador += 1
# #print(contador)

def create_tracer_VTU(fileNumber, prediction, networkName):
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
    filePath = folderPath + '/LSBU_' + str(fileNumber) + '.vtu'
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)

    ug.AddScalarField('Prediction' + networkName, output)
    
    saveFolderPath = '/vol/bitbucket/ja819/Fluids Dataset/predictions'
    saveFilePath = saveFolderPath + '/' + networkName + '_' + str(fileNumber) + '.vtu'

    ug.Write(saveFilePath)
