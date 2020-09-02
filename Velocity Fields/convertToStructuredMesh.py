##############################
# Convert to structured mesh #
##############################

## Dataset ranges are as follows
#   X Range: -359.685 to 359.685 (delta: 719.369)
#   Y Range: -338.124 to 338.124 (delta: 676.247)
#   Z Range: 0.2 to 250 (delta: 249.8)

import numpy as np
import sys
import vtktools
import pyvista as pv
from Variables import defaultFilePath

def get_structured_velocity(fileNumber):
    sys.path.append('fluidity-master')

    fileName = defaultFilePath + '/small3DLSBU/LSBU_' + str(fileNumber) + '.vtu'
    mesh = pv.read(fileName)

    size = 64
    x = np.linspace(-359.69, 359.69, size)
    y = np.linspace(-338.13, 338.13, size)
    z = np.linspace(0.2, 250, size)
    x, y, z = np.meshgrid(x, y, z)

    grid = pv.StructuredGrid(x, y, z)
    result = grid.interpolate(mesh, radius=20.)
    p = result.point_arrays['Velocity']
    p = p.transpose()
    return p


def convert_to_structured(data):
    sys.path.append('fluidity-master')

    fileName = defaultFilePath + '/small3DLSBU/LSBU_0.vtu'
    mesh = pv.read(fileName)

    size = 64
    x = np.linspace(-359.69, 359.69, size)
    y = np.linspace(-338.13, 338.13, size)
    z = np.linspace(0.2, 250, size)
    x, y, z = np.meshgrid(x, y, z)

    grid = pv.StructuredGrid(x, y, z)
    result = grid.interpolate(mesh, radius=20.)
    result.point_arrays['Velocity'] = data   

    foo = mesh.copy()
    foo.clear_arrays()
    result2 = foo.sample(result)

    p = result2.point_arrays['Velocity']

    return p

# #######################################
# # Convert existing data to Structured #
# #######################################
# for fileNumber in range(989):

#     sys.path.append('fluidity-master')
#     fileName = '/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU/LSBU_' + str(fileNumber) + '.vtu'
#     mesh = pv.read(fileName)

#     size = 64
#     x = np.linspace(-359.69, 359.69, size)
#     y = np.linspace(-338.13, 338.13, size)
#     z = np.linspace(0.2, 250, size)
#     x, y, z = np.meshgrid(x, y, z)

#     grid = pv.StructuredGrid(x, y, z)
#     result = grid.interpolate(mesh, radius=20.)

#     result.save('/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU-Structured/LSBU_' + str(fileNumber) + '.vtk')
#     print(fileNumber)
