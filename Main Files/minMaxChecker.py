import numpy as np
import sys
import vtktools
import pyvista as pv
from Variables import x_min, x_max
from Norm import normalise

sys.path.append('fluidity-master')
print("Program running....")

arrayOfMax = np.zeros(998)
arrayOfMin = np.zeros(998)

for i in range(988+1):
    filename = '/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU/LSBU_' + str(i) + '.vtu'
    ug = vtktools.vtu(filename)
    ug.GetFieldNames()

    # Read the values of the tracers and copy into a vector named p
    p = ug.GetScalarField('Tracer')
    #p = normalise(p, x_min, x_max)
    arrayOfMax[i] = p.max()
    arrayOfMin[i] = p.min()

arrayOfMin = np.sort(arrayOfMin)
print(arrayOfMin)

#print(arrayOfMax.max())
#print("")
#print(arrayOfMin.min())