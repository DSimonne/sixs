import numpy as np
from scipy.linalg import norm
import matplotlib
#import csv
#from mayavi import mlab
#import matplotlib
import math
matplotlib.use('qt5agg')
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import glob

################
# open vtk file#
################
#http://forrestbao.blogspot.com/2011/12/reading-vtk-files-in-python-via-python.html


class Facets(object):
 	"""
 		Import and stores data output of facet analyzer plugin for further analysis
	"""
 	def __init__(self, scan, pathdir):
 		super(Facets, self).__init__()
 		self.scan = scan
 		self.pathdir = pathdir
 		self.pathsave = pathdir + "facets_analysis"

 		# Path to data 
		try:
			filename = glob.glob(pathdir + "*" scan + "*.vtk")[0]
		except Exception as e:
			raise e

		self.lattice = 3.912
		self.hkl = [1,1,1]
		self.hkls = ' '.join(str(e) for e in hkl)
		self.planar_dist = lattice/np.sqrt(hkl[0]**2+hkl[1]**2+hkl[2]**2)
		self.disp_range = planar_dist/6
		self.strain_range = 0.001
		self.disp_range_avg = 0.2
		self.strain_range_avg = 0.0005
		self.ref_normal = hkl/norm(hkl)
		self.ref_string = ' '.join(str('{:.0f}'.format(e)) for e in ref_normal)
		self.comment = ''

		self.rotate_particle = True
		self.fixed_reference = True
		self.show_fig = True

	def set_rotation_matrix(u, v, w, u0 = [1,1,1]/np.sqrt(3), v0 = [-1,-1,2]/np.sqrt(6), w0 = [1,-1,0]/np.sqrt(2))
		"""Defining the rotation matrix"""
		self.u0 = u0
		self.v0 = v0
		self.w0 = w0

		self.u = u
		self.v = v
		self.w = np.cross(u/np.linalg.norm(u),v/np.linalg.norm(v))

		if self.w0 == np.cross(self.u0, self.v0):
			print("We verified that u0.v0=w0")
		else:
			print("Non orthogonal frame")

		self.u1 = u/np.linalg.norm(u)
		self.v1 = v/np.linalg.norm(v)
		self.w1 = w/np.linalg.norm(w)

		# transformation matrix
		self.tensor0 = np.array([u0, v0, w0])
		self.tensor1 = np.array([u1, v1, w1])
		self.invb = np.linalg.inv(self.tensor1)
		self.M_rot = np.dot(np.transpose(self.tensor0), np.transpose(self.invb))

	def test_vector(vec):
		"""`vec` needs to be an (1, 3) array, e.g. np.array([-0.833238, -0.418199, -0.300809])"""
		try:
			print(np.dot(self.M_rot, vec/np.linalg.norm(vec)))
		except:
			print("You need to define the rotation matrix before")

	def load_vtk():
		# Load VTK file
		if not os.path.exists(pathsave):
		    os.makedirs(pathsave)

		reader = vtk.vtkGenericDataObjectReader()
		reader.SetFileName(filename)
		reader.ReadAllScalarsOn()
		reader.ReadAllVectorsOn()
		reader.ReadAllTensorsOn()
		reader.Update()
		vtkdata = reader.GetOutput()

		# Get point data
		pointData = vtkdata.GetPointData()
		print("Number of points = %s" % str(vtkdata.GetNumberOfPoints()))
		print("Number of cells = %s" % str(vtkdata.GetNumberOfCells()))

		vtk_data = {}
		vtk_data['disp'] = np.zeros(vtkdata.GetNumberOfPoints())
		vtk_data['strain'] = np.zeros(vtkdata.GetNumberOfPoints())
		vtk_data['x'] = np.zeros(vtkdata.GetNumberOfPoints())
		vtk_data['y'] = np.zeros(vtkdata.GetNumberOfPoints())
		vtk_data['z'] = np.zeros(vtkdata.GetNumberOfPoints())

		# pointData.GetArrayName(1) # to get the name of the array
		# get the positions of the points-voxels // vtkdata.GetPoint(0)[0] or vtkdata.GetPoint(0)
		for i in range(vtkdata.GetNumberOfPoints()):
		    vtk_data['x'][i] = vtkdata.GetPoint(i)[0]
		    vtk_data['y'][i] = vtkdata.GetPoint(i)[1]
		    vtk_data['z'][i] = vtkdata.GetPoint(i)[2]
		    vtk_data['strain'][i] = pointData.GetArray('strain').GetValue(i)
		    vtk_data['disp'][i] = pointData.GetArray('disp').GetValue(i)


		# =============================================================================
		# Get cell data
		# =============================================================================
		    
		cellData = vtkdata.GetCellData()
		vtk_data['FacetProbabilities'] = np.zeros(vtkdata.GetNumberOfCells())
		vtk_data['FacetIds'] = np.zeros(vtkdata.GetNumberOfCells())
		vtk_data['x0'] = np.zeros(vtkdata.GetNumberOfCells())
		vtk_data['y0'] = np.zeros(vtkdata.GetNumberOfCells())
		vtk_data['z0'] = np.zeros(vtkdata.GetNumberOfCells())

		for i in range(vtkdata.GetNumberOfCells()):
		    vtk_data['FacetProbabilities'][i] = cellData.GetArray('FacetProbabilities').GetValue(i)
		    vtk_data['FacetIds'][i] = cellData.GetArray('FacetIds').GetValue(i)
		    vtk_data['x0'][i] = vtkdata.GetCell(i).GetPointId(0)
		    vtk_data['y0'][i] = vtkdata.GetCell(i).GetPointId(1)
		    vtk_data['z0'][i] = vtkdata.GetCell(i).GetPointId(2)

		nb_Facet = int(max(vtk_data['FacetIds']))
		print("Number of facets = %s" % str(nb_Facet))


def extract_facet(facet_id, vtk_data):
    """
    Extract data from one facet, [x, y, z], strain, displacement and their means
    """
    ind_Facet = []
    for i in range(len(vtk_data['FacetIds'])):
        if (int(vtk_data['FacetIds'][i]) == facet_id):
            ind_Facet.append(vtk_data['x0'][i])
            ind_Facet.append(vtk_data['y0'][i])
            ind_Facet.append(vtk_data['z0'][i])

    ind_Facet_new = list(set(ind_Facet))
    results = {}
    results['x'], results['y'], results['z'] = np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new))
    results['strain'], results['disp'] = np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new))

    for j in range(len(ind_Facet_new)):
        results['x'][j] = vtk_data['x'][int(ind_Facet_new[j])]
        results['y'][j] = vtk_data['y'][int(ind_Facet_new[j])]
        results['z'][j] = vtk_data['z'][int(ind_Facet_new[j])]
        results['strain'][j] = vtk_data['strain'][int(ind_Facet_new[j])]
        results['disp'][j] = vtk_data['disp'][int(ind_Facet_new[j])]
    results['strain_mean'] = np.mean(results['strain'])
    results['strain_std'] = np.std(results['strain'])
    results['disp_mean'] = np.mean(results['disp'])
    results['disp_std'] = np.std(results['disp'])
    return results
