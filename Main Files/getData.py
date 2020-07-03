import numpy as np
import sys
import vtktools
import pyvista as pv

def get_tracer(fileNumber):
    """
    Used to get the Tracers as a numpy array corresponding to a vtu file from the Fluids dataset 
    :param fileNumber: int or string
        Used to identify which vtu file to return
        Values are between 0 and 988
    :return: numpy array
        Tracers are returned as numpy array
    """
    #fileNumber = 20
    filePath = 'E:\MSc Individual Project\Fluids Dataset\small3DLSBU\LSBU_' + str(fileNumber) + '.vtu'
    sys.path.append('fluidity-master')
    ug = vtktools.vtu(filePath)
    ug.GetFieldNames()
    p = ug.GetScalarField('Tracer')

    # Convert p into 1 x N array
    p = np.array([p[:]])
    return p

