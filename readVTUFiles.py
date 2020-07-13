import numpy as np
import sys
import vtktools
import pyvista as pv

sys.path.append('fluidity-master')
print("Program running....")
# Check all timesteps exists
dicOfSizes = set()
# for i in range(988+1):
#     filename = 'E:\MSc Individual Project\Fluids Dataset\small3DLSBU\LSBU_' + str(i) + '.vtu'
#     ug = vtktools.vtu(filename)
#     ug.GetFieldNames()

#     # Read the values of the tracers and copy into a vector named p
#     p = ug.GetScalarField('Tracer')
#     print(str(i) + ":")
#     #print(p)
#     #print("The length of p is ", len(p))
#     dicOfSizes.add(len(p)) 

# print(dicOfSizes)

#Print out a given LSBU
# ug = vtktools.vtu('E:\MSc Individual Project\Fluids Dataset\small3DLSBU\LSBU_20.vtu')
ug = vtktools.vtu('/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU/LSBU_20.vtu')

ug.GetFieldNames()
p = ug.GetScalarField('Tracer')
#p = ug.GetVectorField('Velocity')
print("The length of p is ", len(p))
print("")
print("p contains the following: ")
print(p)
print("The type of p is ", type(p))
print("The shape of p is ", p.shape)


#mesh = pv.read('/vol/bitbucket/ja819/Fluids Dataset/small3DLSBU/LSBU_19.vtu')
#cpos = mesh.plot()

# # check each time step exists
# # for loop over it all?
# # ParaView - external program to view the vtu files
# # PyVista - python library to view the vtu files