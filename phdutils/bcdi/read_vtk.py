import numpy as np
from scipy.linalg import norm
import matplotlib
import math
#matplotlib.use('qt5agg')
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import glob
import pandas as pd
import pickle

################
# open vtk file#
################
#http://forrestbao.blogspot.com/2011/12/reading-vtk-files-in-python-via-python.html


class Facets(object):
	"""
		Import and stores data output of facet analyzer plugin for further analysis

		Since the datasets are not that big, we still handle them with pandas, otherwise we should probably keep them on disk in a hdf5

		vtk file in Sxxxx/postprocessing

		Analysis output in Sxxxx/postprocessing/facet_analysis
	"""

	def __init__(self, pathdir, filename):
		super(Facets, self).__init__()
		self.pathsave = pathdir + "facets_analysis/"
		self.filename = pathdir + filename

		self.lattice = 3.912
		self.hkl = [1,1,1]
		self.hkls = ' '.join(str(e) for e in self.hkl)
		self.planar_dist = self.lattice/np.sqrt(self.hkl[0]**2+self.hkl[1]**2+self.hkl[2]**2)
		self.disp_range = self.planar_dist/6
		self.strain_range = 0.001
		self.disp_range_avg = 0.2
		self.strain_range_avg = 0.0005
		self.ref_normal = self.hkl/norm(self.hkl)
		self.ref_string = ' '.join(str('{:.0f}'.format(e)) for e in self.ref_normal)
		self.comment = ''

		self.rotate_particle = True
		self.fixed_reference = True
		self.show_fig = True

		self.cmap = "viridis"


	def set_rotation_matrix(self, u0, v0, w0, u, v):
		"""Defining the rotation matrix"""
		self.u0 = u0
		self.v0 = v0
		self.w0 = w0

		print("Cross product of u0 and v0:", np.cross(self.u0, self.v0))

		self.u = u
		self.v = v
		self.w = np.cross(u/np.linalg.norm(u),v/np.linalg.norm(v))

		self.u1 = self.u/np.linalg.norm(self.u)
		self.v1 = self.v/np.linalg.norm(self.v)
		self.w1 = self.w/np.linalg.norm(self.w)

		# transformation matrix
		self.tensor0 = np.array([self.u0, self.v0, self.w0])
		self.tensor1 = np.array([self.u1, self.v1, self.w1])
		self.invb = np.linalg.inv(self.tensor1)
		self.M_rot = np.dot(np.transpose(self.tensor0), np.transpose(self.invb))


	def test_vector(self, vec):
		"""`vec` needs to be an (1, 3) array, e.g. np.array([-0.833238, -0.418199, -0.300809])"""
		try:
			print(np.dot(self.M_rot, vec/np.linalg.norm(vec)))
		except:
			print("You need to define the rotation matrix before")


	def extract_facet(self, facet_id, plot = False):
		"""
		Extract data from one facet, [x, y, z], strain, displacement and their means, also plots it
		"""
		ind_Facet = []
		for i in range(len(self.vtk_data['FacetIds'])):
		    if (int(self.vtk_data['FacetIds'][i]) == facet_id):
		        ind_Facet.append(self.vtk_data['x0'][i])
		        ind_Facet.append(self.vtk_data['y0'][i])
		        ind_Facet.append(self.vtk_data['z0'][i])

		ind_Facet_new = list(set(ind_Facet))
		results = {}
		results['x'], results['y'], results['z'] = np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new))
		results['strain'], results['disp'] = np.zeros(len(ind_Facet_new)), np.zeros(len(ind_Facet_new))

		for j in range(len(ind_Facet_new)):
		    results['x'][j] = self.vtk_data['x'][int(ind_Facet_new[j])]
		    results['y'][j] = self.vtk_data['y'][int(ind_Facet_new[j])]
		    results['z'][j] = self.vtk_data['z'][int(ind_Facet_new[j])]
		    results['strain'][j] = self.vtk_data['strain'][int(ind_Facet_new[j])]
		    results['disp'][j] = self.vtk_data['disp'][int(ind_Facet_new[j])]
		results['strain_mean'] = np.mean(results['strain'])
		results['strain_std'] = np.std(results['strain'])
		results['disp_mean'] = np.mean(results['disp'])
		results['disp_std'] = np.std(results['disp'])

		# plot single result
		if plot:
			fig = plt.figure("Facets")
			ax = fig.add_subplot(projection='3d')

			ax.scatter(
				self.vtk_data['x'],
				self.vtk_data['y'],
				self.vtk_data['z'], 
				s=0.2, 
				antialiased=True, 
				depthshade=True)

			ax.scatter(
				results['x'],
				results['y'],
				results['z'],
				s=50,
				c = results['strain'], 
				cmap = self.cmap,  
				vmin = -0.025, 
				vmax = 0.025, 
				antialiased=True, 
				depthshade=True)
			plt.show()

		return results


	def load_vtk(self):
		# Load VTK file
		if not os.path.exists(self.pathsave):
		    os.makedirs(self.pathsave)

		reader = vtk.vtkGenericDataObjectReader()
		reader.SetFileName(self.filename)
		reader.ReadAllScalarsOn()
		reader.ReadAllVectorsOn()
		reader.ReadAllTensorsOn()
		reader.Update()
		vtkdata = reader.GetOutput()


		# Get point data
		pointData = vtkdata.GetPointData()
		print("Number of points = %s" % str(vtkdata.GetNumberOfPoints()))
		print("Number of cells = %s" % str(vtkdata.GetNumberOfCells()))

		self.vtk_data = {}
		self.vtk_data['disp'] = np.zeros(vtkdata.GetNumberOfPoints())
		self.vtk_data['strain'] = np.zeros(vtkdata.GetNumberOfPoints())
		self.vtk_data['x'] = np.zeros(vtkdata.GetNumberOfPoints())
		self.vtk_data['y'] = np.zeros(vtkdata.GetNumberOfPoints())
		self.vtk_data['z'] = np.zeros(vtkdata.GetNumberOfPoints())

		# pointData.GetArrayName(1) # to get the name of the array
		# get the positions of the points-voxels // vtkdata.GetPoint(0)[0] or vtkdata.GetPoint(0)
		for i in range(vtkdata.GetNumberOfPoints()):
		    self.vtk_data['x'][i] = vtkdata.GetPoint(i)[0]
		    self.vtk_data['y'][i] = vtkdata.GetPoint(i)[1]
		    self.vtk_data['z'][i] = vtkdata.GetPoint(i)[2]
		    self.vtk_data['strain'][i] = pointData.GetArray('strain').GetValue(i)
		    self.vtk_data['disp'][i] = pointData.GetArray('disp').GetValue(i)


		# Get cell data
		cellData = vtkdata.GetCellData()
		self.vtk_data['FacetProbabilities'] = np.zeros(vtkdata.GetNumberOfCells())
		self.vtk_data['FacetIds'] = np.zeros(vtkdata.GetNumberOfCells())
		self.vtk_data['x0'] = np.zeros(vtkdata.GetNumberOfCells())
		self.vtk_data['y0'] = np.zeros(vtkdata.GetNumberOfCells())
		self.vtk_data['z0'] = np.zeros(vtkdata.GetNumberOfCells())

		for i in range(vtkdata.GetNumberOfCells()):
		    self.vtk_data['FacetProbabilities'][i] = cellData.GetArray('FacetProbabilities').GetValue(i)
		    self.vtk_data['FacetIds'][i] = cellData.GetArray('FacetIds').GetValue(i)
		    self.vtk_data['x0'][i] = vtkdata.GetCell(i).GetPointId(0)
		    self.vtk_data['y0'][i] = vtkdata.GetCell(i).GetPointId(1)
		    self.vtk_data['z0'][i] = vtkdata.GetCell(i).GetPointId(2)

		self.nb_facets = int(max(self.vtk_data['FacetIds']))
		print("Number of facets = %s" % str(self.nb_facets))


		# Get means
		facet = np.arange(1, int(self.nb_facets) + 1, 1)
		strain_mean = np.zeros(len(facet)) # stored later in field data
		strain_std = np.zeros(len(facet)) # stored later in field data
		disp_mean = np.zeros(len(facet)) # stored later in field data
		disp_std = np.zeros(len(facet)) # stored later in field data
		self.strain_mean_facets=[]
		self.disp_mean_facets=[]

		for ind in np.arange(1, int(self.nb_facets) + 1, 1):
		    print("Facet = %d" % ind)
		    results = self.extract_facet(ind, plot = False)
		    strain_mean[ind-1] = results['strain_mean']
		    strain_std[ind-1] = results['strain_std']
		    disp_mean[ind-1] = results['disp_mean']
		    disp_std[ind-1] = results['disp_std']


		# Get field data
		self.field_data = {}
		fieldData = vtkdata.GetFieldData()

		self.field_data['facet'] = facet
		self.field_data['strain_mean'] = strain_mean
		self.field_data['strain_std'] = strain_std
		self.field_data['disp_mean'] = disp_mean
		self.field_data['disp_std'] = disp_std

		self.field_data['n0'] = np.zeros(self.nb_facets)
		self.field_data['n1'] = np.zeros(self.nb_facets)
		self.field_data['n2'] = np.zeros(self.nb_facets)
		self.field_data['FacetIds'] = np.zeros(self.nb_facets)
		self.field_data['absFacetSize'] = np.zeros(self.nb_facets)
		self.field_data['interplanarAngles'] = np.zeros(self.nb_facets)
		self.field_data['relFacetSize'] = np.zeros(self.nb_facets)

		for i in range(self.nb_facets):
		    self.field_data['n0'][i] = fieldData.GetArray('facetNormals').GetValue(3*i)
		    self.field_data['n1'][i] = fieldData.GetArray('facetNormals').GetValue(3*i+1)
		    self.field_data['n2'][i] = fieldData.GetArray('facetNormals').GetValue(3*i+2)
		    self.field_data['FacetIds'][i] = fieldData.GetArray('FacetIds').GetValue(i)
		    self.field_data['absFacetSize'][i] = fieldData.GetArray('absFacetSize').GetValue(i)
		    self.field_data['interplanarAngles'][i] = fieldData.GetArray('interplanarAngles').GetValue(i)
		    self.field_data['relFacetSize'][i] = fieldData.GetArray('relFacetSize').GetValue(i)


		# Get self.normals
		self.normals = np.zeros((self.nb_facets, 3))

		self.field_data['interplanarAngles']=np.delete(self.field_data['interplanarAngles'], self.nb_facets-1,0)
		self.field_data['interplanarAngles']=np.append(np.zeros(1), self.field_data['interplanarAngles'],axis=0)

		for i in range(self.nb_facets):
		    self.normals[i]= np.array([self.field_data['n0'][i], self.field_data['n1'][i], self.field_data['n2'][i]])
		    

		# Rotate particle if required
		if self.rotate_particle:
		    for i in range(self.nb_facets):
		        self.normals[i] = np.dot(self.M_rot, self.normals[i])


		# Interplanar angle fixed reference
		if self.fixed_reference:
		    for i in range(self.nb_facets):
		        self.field_data['interplanarAngles'][i] = math.acos(np.dot(self.ref_normal, self.normals[i]/norm(self.normals[i])))*180./np.pi


		self.legend = []
		for i in range(self.nb_facets):
		    #legend = legend + [' '.join(str('{:.1f}'.format(e)) for e in normals[i])]
		    #legend = legend + [' '.join(str('{:.0f}'.format(e)) for e in normals[i]*2)]
		    self.legend = self.legend + [' '.join(str('{:.2f}'.format(e)) for e in self.normals[i])]

		self.field_data = pd.DataFrame(self.field_data)


	def plot_strain(self, figsize = (12, 10)):

		# 3D strain
		fig_name = 'strain_3D_' + self.hkls + '_' + str(self.strain_range)
		fig = plt.figure(figsize = figsize)
		ax = fig.add_subplot(projection='3d')

		for ind in range(1, self.nb_facets):
		    results = self.extract_facet(ind, plot = False)

		    p = ax.scatter(
		    	results['x'], 
		    	results['y'],
		    	results['z'], 
		    	s=50, 
		    	c = results['strain'],
		    	cmap = self.cmap,  
		    	vmin = -self.strain_range, 
		    	vmax = self.strain_range, 
		    	antialiased=True, 
		    	depthshade=True)

		fig.colorbar(p)
		# ax.view_init(elev=-64, azim=94)
		plt.title("Strain for each voxel")
		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')
		plt.show()

		# Average strain
		fig_name = 'strain_3D_avg_' + self.hkls + '_' + str(self.strain_range_avg) 
		fig = plt.figure(figsize = figsize)
		ax = fig.add_subplot(projection='3d')

		for ind in range(1, self.nb_facets):
		    results = self.extract_facet(ind, plot = False)

		    strain_mean_facet = np.zeros(results['strain'].shape)
		    strain_mean_facet.fill(results['strain_mean'])
		    self.strain_mean_facets=np.append(self.strain_mean_facets, strain_mean_facet,axis=0)

		    p = ax.scatter(
		    	results['x'], 
		    	results['y'],
		    	results['z'], 
		    	s=50, 
		    	c = strain_mean_facet,
		    	cmap = self.cmap,  
		    	vmin = -self.strain_range_avg, 
		    	vmax = self.strain_range_avg, 
		    	antialiased=True, 
		    	depthshade=True)

		fig.colorbar(p)
		plt.title("Mean strain per facet")
		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')
		plt.show()


	def plot_displacement(self, figsize = (12,10)):

		# 3D displacement
		fig_name = 'disp_3D_' + self.hkls + '_' + str(self.disp_range)
		fig = plt.figure(figsize = figsize)
		ax = fig.add_subplot(projection='3d')

		for ind in range(1, self.nb_facets):
		    results = self.extract_facet(ind, plot = False)

		    p = ax.scatter(
		    	results['x'], 
		    	results['y'],
		    	results['z'], 
		    	s=50, 
		    	c = results['disp'],
		    	cmap = self.cmap,  
		    	vmin = -self.disp_range, 
		    	vmax = self.disp_range, 
		    	antialiased=True, 
		    	depthshade=True)

		ax.view_init(elev=-64, azim=94)
		fig.colorbar(p)
		plt.title("Displacement for each voxel")
		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')


		# Average disp
		fig_name = 'disp_3D_avg_' + self.hkls + '_' + str(self.disp_range_avg)
		fig = plt.figure(figsize = figsize)
		ax = fig.add_subplot(projection='3d')

		for ind in range(1, self.nb_facets):
		    results = self.extract_facet(ind, plot = False)

		    disp_mean_facet = np.zeros(results['disp'].shape)
		    disp_mean_facet.fill(results['disp_mean'])
		    self.disp_mean_facets=np.append(self.strain_mean_facets, disp_mean_facet,axis=0)

		    p = ax.scatter(
		    	results['x'],
		    	results['y'],
		    	results['z'], 
		    	s=50, 
		    	c = disp_mean_facet,
		    	cmap = self.cmap,  
		    	vmin = -self.disp_range_avg/2, 
		    	vmax = self.disp_range_avg/2, 
		    	antialiased=True, 
		    	depthshade=True)

		ax.view_init(elev=-64, azim=94)
		fig.colorbar(p)
		plt.title("Mean displacement per facet")
		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')


	def evolution_curves(self):

		# 1D plot: average displacement vs facet index
		fig_name = 'avg_disp_vs_facet_id_' + self.hkls + self.comment
		plt.figure(figsize = (10,6))
		for i in range(self.nb_facets):
		    plt.errorbar(self.field_data['facet'][i], self.field_data['disp_mean'][i], self.field_data['disp_std'][i], fmt='o', label = self.legend[i])
		    plt.xlabel('Facet index')
		    plt.ylabel('Average retrieved displacement')
		    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
		    plt.grid()
		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')


		# 1D plot: average strain vs facet index
		fig_name = 'avg_strain_vs_facet_id_' + self.hkls + self.comment
		plt.figure(figsize = (10,6))
		for i in range(self.nb_facets):
		    plt.errorbar(self.field_data['facet'][i], self.field_data['strain_mean'][i], self.field_data['strain_std'][i], fmt='o', label = self.legend[i])
		    plt.xlabel('Facet index')
		    plt.ylabel('Average retrieved strain')
		    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
		    plt.grid()
		plt.savefig(self.pathsave + fig_name +'.png', bbox_inches = 'tight')


		# Subplots
		fig_name = 'disp_strain_size_vs_angle_planes_' + self.hkls + self.comment
		plt.figure(figsize = (10,12))

		# 1D plot: average strain vs angle with respect to the reference facet
		plt.subplot(3,1,1)
		for i in range(self.nb_facets):
		    plt.errorbar(self.field_data['interplanarAngles'][i], self.field_data['disp_mean'][i], self.field_data['disp_std'][i], fmt='o',capsize=2, label = self.legend[i])
		    #plt.xlabel('Angle (deg.)')6
		    plt.ylabel('Retrieved <disp> (Angstroms)')
		    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=5, fancybox=True, shadow=True)
		    plt.grid()

		# 1D plot: average displacement vs angle with respect to the reference facet
		plt.subplot(3,1,2)
		for i in range(self.nb_facets):
		    plt.errorbar(self.field_data['interplanarAngles'][i], self.field_data['strain_mean'][i], self.field_data['strain_std'][i], fmt='o',capsize=2, label = self.legend[i])
		    #plt.xlabel('Angle (deg.)')
		    plt.ylabel('Retrieved <strain>')
		    plt.grid()

		# 1D plot: relative facet size vs angle with respect to the reference facet
		plt.subplot(3,1,3)
		for i in range(self.nb_facets):
		    plt.plot(self.field_data['interplanarAngles'][i], self.field_data['relFacetSize'][i],'o', label = self.legend[i])
		    plt.xlabel('Angle (deg.)')
		    plt.ylabel('Relative facet size')
		    plt.grid()
		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')


		fig_name = 'disp_strain_size_vs_angle_planes_' + self.hkls + self.comment 
		plt.figure(figsize = (10,12))
		plt.subplot(3,1,1)
		for i in range(self.nb_facets):
		    lx, ly, lz = float(self.legend[i].split()[0]),float(self.legend[i].split()[1]),float(self.legend[i].split()[2])
		    if (lx>=0 and ly>=0):
		        fmt='o'
		    if (lx>=0 and ly<=0):
		        fmt='d'
		    if (lx<=0 and ly>=0):
		        fmt='s'
		    if (lx<=0 and ly<=0):
		        fmt = '+'
		    plt.errorbar(self.field_data['interplanarAngles'][i], self.field_data['disp_mean'][i], self.field_data['disp_std'][i], fmt=fmt,capsize=2, label = self.legend[i])
		    #plt.xlabel('Angle (deg.)')6
		    plt.ylabel('Retrieved <disp> (Angstroms)')
		    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=5, fancybox=True, shadow=True)
		    #plt.legend(bbox_to_anchor=(1, 0), loc='upper left', borderaxespad=-0.1)
		    plt.grid()
		plt.subplot(3,1,2)
		for i in range(self.nb_facets):
		    lx, ly, lz = float(self.legend[i].split()[0]),float(self.legend[i].split()[1]),float(self.legend[i].split()[2])
		    if (lx>=0 and ly>=0):
		        fmt='o'
		    if (lx>=0 and ly<=0):
		        fmt='d'
		    if (lx<=0 and ly>=0):
		        fmt='s'
		    if (lx<=0 and ly<=0):
		        fmt = '+'
		    plt.errorbar(self.field_data['interplanarAngles'][i], self.field_data['strain_mean'][i], self.field_data['strain_std'][i], fmt=fmt,capsize=2, label = self.legend[i])
		    #plt.xlabel('Angle (deg.)')
		    plt.ylabel('Retrieved <strain>')
		    plt.grid()
		plt.subplot(3,1,3)
		for i in range(self.nb_facets):
		    plt.plot(self.field_data['interplanarAngles'][i], self.field_data['relFacetSize'][i],'o', label = self.legend[i])
		    plt.xlabel('Angle (deg.)')
		    plt.ylabel('Relative facet size')
		    plt.grid()
		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')


	def save_data(self, filename):
		# Save field data
		self.field_data.to_csv(filename)

	def pickle(self, filename):
		# Use the pickle module to save the classes
		try:
		    with open(filename, 'wb') as f:
		        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		except PermissionError:
		    print("""Permission denied, You cannot save this file because you are not its creator. The changes are updated for this session and you can still plot figures but once you exit the program, all changes will be erased.""")
		    pass

	@staticmethod
	def unpickle(prompt):
		"""Use the pickle module to load the classes
		"""

		with open(f"{prompt}", 'rb') as f:
		    return pickle.load(f)