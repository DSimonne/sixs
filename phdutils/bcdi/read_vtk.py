import numpy as np
import matplotlib
#matplotlib.use('qt5agg')
import vtk
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import pickle

################
# open vtk file#
################
#http://forrestbao.blogspot.com/2011/12/reading-vtk-files-in-python-via-python.html


# Extract strain at facets
# Need a vtk file extracted from the FacetAnalyser plugin of ParaView (information: point data, cell data and field data)
# mrichard@esrf.fr & maxime.dupraz@esrf.fr

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
		self.strain_range = 0.001
		self.disp_range_avg = 0.2
		self.disp_range = 0.35
		self.strain_range_avg = 0.0005
		self.comment = ''

		self.title_fontsize = 24
		self.axes_fontsize = 18
		self.legend_fontsize = 11
		self.ticks_fontsize = 14

		self.cmap = "viridis"

		# Load the data
		self.load_vtk()


	def load_vtk(self):
		"""Load VTK file
			In paraview, the facets have an index that starts at 1, but here we start at 0 because it's python
		"""
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
		try:
			pointData = vtkdata.GetPointData()
			print("Loading data...")
		except AttributeError:
			print("This file does not exist or is not right.")

		print("Number of points = %s" % str(vtkdata.GetNumberOfPoints()))
		print("Number of cells = %s" % str(vtkdata.GetNumberOfCells()))

		self.vtk_data = {}

		self.vtk_data['x'] = [vtkdata.GetPoint(i)[0] for i in range(vtkdata.GetNumberOfPoints())]
		self.vtk_data['y'] = [vtkdata.GetPoint(i)[1] for i in range(vtkdata.GetNumberOfPoints())]
		self.vtk_data['z'] = [vtkdata.GetPoint(i)[2] for i in range(vtkdata.GetNumberOfPoints())]
		self.vtk_data['strain'] = [pointData.GetArray('strain').GetValue(i) for i in range(vtkdata.GetNumberOfPoints())]
		self.vtk_data['disp'] = [pointData.GetArray('disp').GetValue(i) for i in range(vtkdata.GetNumberOfPoints())]


		# Get cell data
		cellData = vtkdata.GetCellData()

		self.vtk_data['facet_probabilities'] = [cellData.GetArray('FacetProbabilities').GetValue(i) for i in range(vtkdata.GetNumberOfCells())]
		self.vtk_data['facet_id'] = [cellData.GetArray('FacetIds').GetValue(i) for i in range(vtkdata.GetNumberOfCells())]
		self.vtk_data['x0'] = [vtkdata.GetCell(i).GetPointId(0) for i in range(vtkdata.GetNumberOfCells())]
		self.vtk_data['y0'] = [vtkdata.GetCell(i).GetPointId(1) for i in range(vtkdata.GetNumberOfCells())]
		self.vtk_data['z0'] = [vtkdata.GetCell(i).GetPointId(2) for i in range(vtkdata.GetNumberOfCells())]

		self.nb_facets = int(max(self.vtk_data['facet_id']))
		print("Number of facets = %s" % str(self.nb_facets))

		# Get means
		facet_indices = np.arange(1, int(self.nb_facets) + 1, 1) # indices from 1 to n_facets

		strain_mean = np.zeros(self.nb_facets) # stored later in field data
		strain_std = np.zeros(self.nb_facets) # stored later in field data
		disp_mean = np.zeros(self.nb_facets) # stored later in field data
		disp_std = np.zeros(self.nb_facets) # stored later in field data

		# For future analysis
		self.strain_mean_facets=[]
		self.disp_mean_facets=[]

		for ind in facet_indices:
		    print("Facet = %d" % ind)
		    results = self.extract_facet(ind, plot = False)
		    strain_mean[ind-1] = results['strain_mean']
		    strain_std[ind-1] = results['strain_std']
		    disp_mean[ind-1] = results['disp_mean']
		    disp_std[ind-1] = results['disp_std']


		# Get field data
		self.field_data = pd.DataFrame()
		fieldData = vtkdata.GetFieldData()

		self.field_data['facet_id'] = [fieldData.GetArray('FacetIds').GetValue(i) for i in range(self.nb_facets)]
		self.field_data['strain_mean'] = strain_mean
		self.field_data['strain_std'] = strain_std
		self.field_data['disp_mean'] = disp_mean
		self.field_data['disp_std'] = disp_std
		self.field_data['n0'] = [fieldData.GetArray('facetNormals').GetValue(3*i) for i in range(self.nb_facets)]
		self.field_data['n1'] = [fieldData.GetArray('facetNormals').GetValue(3*i+1) for i in range(self.nb_facets)]
		self.field_data['n2'] = [fieldData.GetArray('facetNormals').GetValue(3*i+2) for i in range(self.nb_facets)]
		self.field_data['c0'] = [fieldData.GetArray('FacetCenters').GetValue(3*i) for i in range(self.nb_facets)]
		self.field_data['c1'] = [fieldData.GetArray('FacetCenters').GetValue(3*i+1) for i in range(self.nb_facets)]
		self.field_data['c2'] = [fieldData.GetArray('FacetCenters').GetValue(3*i+2) for i in range(self.nb_facets)]
		self.field_data['interplanar_angles'] = [fieldData.GetArray('interplanarAngles').GetValue(i) for i in range(self.nb_facets)]
		self.field_data['abs_facet_size'] = [fieldData.GetArray('absFacetSize').GetValue(i) for i in range(self.nb_facets)]
		self.field_data['rel_facet_size'] = [fieldData.GetArray('relFacetSize').GetValue(i) for i in range(self.nb_facets)]

		self.field_data = self.field_data.astype({'facet_id': np.int8})

		# Get normals
		# Don't use array index but facet number in case we sort the dataframe !!
		normals = {f"facet_{row.facet_id}":  np.array([row['n0'], row['n1'], row['n2']]) for j, row in self.field_data.iterrows()}

		# Update legend
		legend = []
		for e in normals.keys():
		    legend = legend + [' '.join(str('{:.2f}'.format(e)) for e in normals[e])]
		self.field_data["legend"] = legend


	def set_rotation_matrix(self, u0, v0, w0, u, v):
		"""Defining the rotation matrix
		u and v should be the vectors perpendicular to two facets
		the rotation matric is then used if the argument rotate_particle is set to true in the load_vtk method"""
		self.u0 = u0
		self.v0 = v0
		self.w0 = w0
		print("Cross product of u0 and v0:", np.cross(self.u0, self.v0))

		self.u = u
		self.v = v
		self.w = np.cross(u/np.linalg.norm(u),v/np.linalg.norm(v))
		print("Cross product of u and v:", np.cross(self.u, self.v))

		self.u1 = self.u/np.linalg.norm(self.u)
		self.v1 = self.v/np.linalg.norm(self.v)
		self.w1 = self.w/np.linalg.norm(self.w)

		# transformation matrix
		self.tensor0 = np.array([self.u0, self.v0, self.w0])
		self.tensor1 = np.array([self.u1, self.v1, self.w1])
		self.inv_tensor1 = np.linalg.inv(self.tensor1)
		self.M_rot = np.dot(np.transpose(self.tensor0), np.transpose(self.inv_tensor1))


	def rotate_particle(self):
		"""Rotate the particle so that the base of the normals to the facets is computed with the new 
		rotation matrix"""

		# Get normals, again to make sure that we have the good ones
		normals = {f"facet_{row.facet_id}":  np.array([row['n0'], row['n1'], row['n2']]) for j, row in self.field_data.iterrows()}

		try:
		    for e in normals.keys():
		        normals[e] = np.dot(self.M_rot, normals[e])
		except:
			print("""You need to define the rotation matrix first if you want to rotate the particle.
				Please choose vectors from the normals in field data""")

		# Save the new normals
		v0, v1, v2 = [], [], []
		for k, v in normals.items():
			# we make sure that we use the same facets !!
			mask = self.field_data["facet_id"] == int(k.split("facet_")[-1])
			self.field_data.loc[mask, 'n0'] = v[0]
			self.field_data.loc[mask, 'n1'] = v[1]
			self.field_data.loc[mask, 'n2'] = v[2]

		# Update legend
		legend = []
		for e in normals.keys():
		    legend = legend + [' '.join(str('{:.2f}'.format(e)) for e in normals[e])]
		self.field_data["legend"] = legend


	def fixed_reference(self, hkl = [1,1,1], plot = True):
		"""Recompute the interplanar angles between each normal and a fixed reference vector"""

		self.hkl = hkl
		self.hkls = ' '.join(str(e) for e in self.hkl)
		self.planar_dist = self.lattice/np.sqrt(self.hkl[0]**2+self.hkl[1]**2+self.hkl[2]**2)
		self.ref_normal = self.hkl/np.linalg.norm(self.hkl)

		# Get normals, again to make sure that we have the good ones
		normals = {f"facet_{row.facet_id}":  np.array([row['n0'], row['n1'], row['n2']]) for j, row in self.field_data.iterrows()}

		# Interplanar angle recomputed from a fixed reference plane, between the experimental facets
		new_angles = []
		for e in normals.keys():
			value = np.rad2deg(
						np.arccos(
							np.dot(self.ref_normal, normals[e]/np.linalg.norm(normals[e]))
							)
						)

			new_angles.append(value)

		# Convert nan to zeros
		mask = np.isnan(new_angles)
		for j, m in enumerate(mask):
		    if m:
		        new_angles[j] = 0

		self.field_data['interplanar_angles'] = new_angles

		# Save angles for indexation, using facets that we should see or usually see on Pt nanoparticles (WK form)
		normals = [[1, 1, 0], [-1, 1, 0], [-1, -1, 0],
		           [1, 0, 0], [-1, 0, 0],
		           [2, 1, 0],
		           [1, 1, 3], [1, -1, 3], [1, -1, -3], [-1, -1, 3], [1, 1, -3], [-1, -1, -3],
		           [1, -1, 1], [-1, 1, -1]
		          ]

		# Stores the theoretical angles between normals
		self.angles = {}
		for n in normals:
		    self.angles[str(n)] = np.rad2deg(
									np.arccos(
										np.dot(self.ref_normal, n/np.linalg.norm(n))
												)
											)

		# Make a plot
		if plot == True:
			fig = plt.figure(figsize = (10, 6))
			ax = fig.add_subplot(1, 1, 1)

			ax.set_title("Interplanar angles between [111] and other possible facets", fontsize = self.title_fontsize)
      
			for norm, (norm_str, angle) in zip(normals, self.angles.items()):
				# add colors ass a fct of multiplicity
			    if [abs(x) for x in norm] == [1, 1, 1]:
			        color = "#7fc97f"
			    if [abs(x) for x in norm] == [1, 1, 0]:
			        color = "#beaed4"
			    if [abs(x) for x in norm] == [1, 0, 0]:
			        color = "#fdc086"
			    if [abs(x) for x in norm] == [2, 1, 0]:
			        color = "#f0027f"
			    if [abs(x) for x in norm] == [1, 1, 3]:
			        color = "#386cb0"
			        
			    ax.scatter(angle, norm_str, color = color)

			# Major ticks every 20, minor ticks every 5
			major_ticks = np.arange(0, 180, 20)
			minor_ticks = np.arange(0, 180, 5)

			ax.set_xticks(major_ticks)
			ax.set_xticks(minor_ticks, minor=True)
			# ax.set_yticks(major_ticks)
			# ax.set_yticks(minor_ticks, minor=True)

			# Or if you want different settings for the grids:
			ax.grid(which='minor', alpha=0.2)
			ax.grid(which='major', alpha=0.5)


	def test_vector(self, vec):
		"""`vec` needs to be an (1, 3) array, e.g. np.array([-0.833238, -0.418199, -0.300809])"""
		try:
			print(np.dot(self.M_rot, vec/np.linalg.norm(vec)))
		except:
			print("You need to define the rotation matrix before")


	def extract_facet(self, facet_id, plot = False, view = [0, 90], output = True):
		"""
		Extract data from one facet, [x, y, z], strain, displacement and their means, also plots it
		"""

		ind_Facet = []
		for i in range(len(self.vtk_data['facet_id'])):
		    if (int(self.vtk_data['facet_id'][i]) == facet_id):
		        ind_Facet.append(self.vtk_data['x0'][i])
		        ind_Facet.append(self.vtk_data['y0'][i])
		        ind_Facet.append(self.vtk_data['z0'][i])
		
		if plot:
			try:
				mask = self.field_data.loc[self.field_data["facet_id"] == facet_id]

				n0 = self.field_data.loc[mask, "n0"]
				n1 = self.field_data.loc[mask, "n1"]
				n2 = self.field_data.loc[mask, "n2"]
				print(f"Facet normal: [{n0}, {n1}, {n2}].")
			except:
				# Not yet defined 
				pass

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
			fig = plt.figure(figsize = (10, 10))
			ax = fig.add_subplot(projection='3d')
			ax.view_init(elev = view[0], azim = view[1])

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

			plt.tick_params(axis='both', which='major', labelsize = self.ticks_fontsize)
			plt.tick_params(axis='both', which='minor', labelsize = self.ticks_fontsize)
			plt.title(f"Strain for facet n°{facet_id}", fontsize = self.title_fontsize)
			plt.tight_layout()
			plt.savefig(f"{self.pathsave}facet_n°{facet_id}.png", bbox_inches='tight')
			plt.show()

		if output:
			return results


	def plot_strain(self, figsize = (12, 10), view = [20, 60], save = True):

		# 3D strain
		fig_name = 'strain_3D_' + self.hkls + self.comment + '_' + str(self.strain_range)
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
		ax.view_init(elev = view[0], azim = view[1])
		plt.title("Strain for each voxel", fontsize = self.title_fontsize)
		ax.tick_params(axis='both', which='major', labelsize = self.ticks_fontsize)
		ax.tick_params(axis='both', which='minor', labelsize = self.ticks_fontsize)

		if save:
			plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')
		plt.show()

		# Average strain
		fig_name = 'strain_3D_avg_' + self.hkls + self.comment + '_' + str(self.strain_range_avg) 
		fig = plt.figure(figsize = figsize)
		ax = fig.add_subplot(projection='3d')

		for ind in range(1, self.nb_facets):
		    results = self.extract_facet(ind, plot = False)

		    strain_mean_facet = np.zeros(results['strain'].shape)
		    strain_mean_facet.fill(results['strain_mean'])
		    self.strain_mean_facets=np.append(self.strain_mean_facets, strain_mean_facet, axis=0)

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
		ax.view_init(elev = view[0], azim = view[1])
		plt.title("Mean strain per facet", fontsize = self.title_fontsize)
		ax.tick_params(axis='both', which='major', labelsize = self.ticks_fontsize)
		ax.tick_params(axis='both', which='minor', labelsize = self.ticks_fontsize)

		if save:
			plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')
		plt.show()


	def plot_displacement(self, figsize = (12,10), view = [20, 60], save = True):

		# 3D displacement
		fig_name = 'disp_3D_' + self.hkls + self.comment + '_' + str(self.disp_range)
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

		fig.colorbar(p)
		ax.view_init(elev = view[0], azim = view[1])
		plt.title("Displacement for each voxel", fontsize = self.title_fontsize)
		ax.tick_params(axis='both', which='major', labelsize = self.ticks_fontsize)
		ax.tick_params(axis='both', which='minor', labelsize = self.ticks_fontsize)

		if save:
			plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')
		plt.show()


		# Average disp
		fig_name = 'disp_3D_avg_' + self.hkls + self.comment + '_' + str(self.disp_range_avg)
		fig = plt.figure(figsize = figsize)
		ax = fig.add_subplot(projection='3d')

		for ind in range(1, self.nb_facets):
		    results = self.extract_facet(ind, plot = False)

		    disp_mean_facet = np.zeros(results['disp'].shape)
		    disp_mean_facet.fill(results['disp_mean'])
		    self.disp_mean_facets=np.append(self.disp_mean_facets, disp_mean_facet,axis=0)

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

		fig.colorbar(p)
		ax.view_init(elev = view[0], azim = view[1])
		plt.title("Mean displacement per facet", fontsize = self.title_fontsize)
		ax.tick_params(axis='both', which='major', labelsize = self.ticks_fontsize)
		ax.tick_params(axis='both', which='minor', labelsize = self.ticks_fontsize)

		if save:
			plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')
		plt.show()


	def evolution_curves(self, ncol = 1):

		# 1D plot: average displacement vs facet index
		fig_name = 'avg_disp_vs_facet_id_' + self.hkls + self.comment
		fig = plt.figure(figsize = (10, 6))
		ax = fig.add_subplot(1, 1, 1)

		# Major x ticks every 5, minor ticks every 1
		major_x_ticks_facet = np.arange(0, self.nb_facets+5, 5)
		minor_x_ticks_facet = np.arange(0, self.nb_facets+5, 1)

		ax.set_xticks(major_x_ticks_facet)
		ax.set_xticks(minor_x_ticks_facet, minor=True)
		plt.xticks(fontsize = self.ticks_fontsize)

		# Major y ticks every 0.5, minor ticks every 0.1
		major_y_ticks_facet = np.arange(-3, 3, 0.5)
		minor_y_ticks_facet = np.arange(-3, 3, 0.1)

		ax.set_yticks(major_y_ticks_facet)
		ax.set_yticks(minor_y_ticks_facet, minor=True)
		plt.yticks(fontsize = self.ticks_fontsize)

		for j, row in self.field_data.iterrows():
		    ax.errorbar(row['facet_id'], row['disp_mean'], row['disp_std'], fmt='o', label = row["legend"])
		
		ax.set_title("Average displacement vs facet index", fontsize = self.title_fontsize)
		ax.set_xlabel('Facet index', fontsize = self.axes_fontsize)
		ax.set_ylabel('Average retrieved displacement', fontsize = self.axes_fontsize)

		ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize = self.legend_fontsize, ncol = ncol)

		ax.grid(which='minor', alpha=0.2)
		ax.grid(which='major', alpha=0.5)

		plt.savefig(self.pathsave + fig_name + '.png', bbox_inches = 'tight')


		# 1D plot: average strain vs facet index
		fig_name = 'avg_strain_vs_facet_id_' + self.hkls + self.comment
		fig = plt.figure(figsize = (10, 6))
		ax = fig.add_subplot(1, 1, 1)

		# Major x ticks every 5, minor ticks every 1
		ax.set_xticks(major_x_ticks_facet)
		ax.set_xticks(minor_x_ticks_facet, minor=True)
		plt.xticks(fontsize = self.ticks_fontsize)

		# Major y ticks every 0.5, minor ticks every 0.1
		major_y_ticks_facet = np.arange(-0.0004, 0.0004, 0.0001)
		minor_y_ticks_facet = np.arange(-0.0004, 0.0004, 0.00005)

		ax.set_yticks(major_y_ticks_facet)
		ax.set_yticks(minor_y_ticks_facet, minor=True)
		plt.yticks(fontsize = self.ticks_fontsize)

		for j, row in self.field_data.iterrows():
		    ax.errorbar(row['facet_id'], row['strain_mean'], row['strain_std'], fmt='o', label = row["legend"])
		
		ax.set_title("Average strain vs facet index", fontsize = self.title_fontsize)
		ax.set_xlabel('Facet index', fontsize = self.axes_fontsize)
		ax.set_ylabel('Average retrieved strain', fontsize = self.axes_fontsize)

		ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize = self.legend_fontsize, ncol = ncol)

		ax.grid(which='minor', alpha=0.2)
		ax.grid(which='major', alpha=0.5)

		plt.savefig(self.pathsave + fig_name +'.png', bbox_inches = 'tight')


		# disp, strain & size vs angle planes, change line style as a fct of the planes indices
		fig_name = 'disp_strain_size_vs_angle_planes_' + self.hkls + self.comment 
		fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex = True, figsize = (10,12))

		plt.xticks(fontsize = self.ticks_fontsize)
		plt.yticks(fontsize = self.ticks_fontsize)

		# Major ticks every 20, minor ticks every 5
		major_x_ticks = np.arange(0, 200, 20)
		minor_x_ticks = np.arange(0, 200, 5)

		ax0.set_xticks(major_x_ticks)
		ax0.set_xticks(minor_x_ticks, minor=True)

		# Major y ticks every 0.5, minor ticks every 0.1
		major_y_ticks = np.arange(-3, 3, 0.5)
		minor_y_ticks = np.arange(-3, 3, 0.1)

		ax0.set_yticks(major_y_ticks)
		ax0.set_yticks(minor_y_ticks, minor=True)

		for j, row in self.field_data.iterrows():
		    lx, ly, lz = float(row.legend.split()[0]),float(row.legend.split()[1]),float(row.legend.split()[2])
		    if (lx>=0 and ly>=0):
		        fmt='o'
		    if (lx>=0 and ly<=0):
		        fmt='d'
		    if (lx<=0 and ly>=0):
		        fmt='s'
		    if (lx<=0 and ly<=0):
		        fmt = '+'
		    ax0.errorbar(row['interplanar_angles'], row['disp_mean'], row['disp_std'], fmt=fmt,capsize=2, label = row["legend"])
		ax0.set_ylabel('Retrieved <disp> (A)', fontsize = self.axes_fontsize)
		ax0.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True, fontsize = self.legend_fontsize+3)
		ax0.grid(which='minor', alpha=0.2)
		ax0.grid(which='major', alpha=0.5)


		# Major ticks every 20, minor ticks every 5
		ax1.set_xticks(major_x_ticks)
		ax1.set_xticks(minor_x_ticks, minor=True)

		# Major y ticks every 0.5, minor ticks every 0.1
		major_y_ticks = np.arange(-0.0004, 0.0004, 0.0001)
		minor_y_ticks = np.arange(-0.0004, 0.0004, 0.00005)
		
		ax1.set_yticks(major_y_ticks)
		ax1.set_yticks(minor_y_ticks, minor=True)

		for j, row in self.field_data.iterrows():
		    lx, ly, lz = float(row.legend.split()[0]),float(row.legend.split()[1]),float(row.legend.split()[2])
		    if (lx>=0 and ly>=0):
		        fmt='o'
		    if (lx>=0 and ly<=0):
		        fmt='d'
		    if (lx<=0 and ly>=0):
		        fmt='s'
		    if (lx<=0 and ly<=0):
		        fmt = '+'
		    ax1.errorbar(row['interplanar_angles'], row['strain_mean'], row['strain_std'], fmt=fmt,capsize=2, label = row["legend"])
		ax1.set_ylabel('Retrieved <strain>', fontsize = self.axes_fontsize)
		ax1.grid(which='minor', alpha=0.2)
		ax1.grid(which='major', alpha=0.5)

		# Major ticks every 20, minor ticks every 5
		ax2.set_xticks(major_x_ticks)
		ax2.set_xticks(minor_x_ticks, minor=True)

		# Major y ticks every 0.5, minor ticks every 0.1
		major_y_ticks = np.arange(-0, 0.3, 0.05)
		minor_y_ticks = np.arange(-0, 0.3, 0.01)
		
		ax2.set_yticks(major_y_ticks)
		ax2.set_yticks(minor_y_ticks, minor=True)

		for j, row in self.field_data.iterrows():
		    lx, ly, lz = float(row.legend.split()[0]),float(row.legend.split()[1]),float(row.legend.split()[2])
		    if (lx>=0 and ly>=0):
		        fmt='o'
		    if (lx>=0 and ly<=0):
		        fmt='d'
		    if (lx<=0 and ly>=0):
		        fmt='s'
		    if (lx<=0 and ly<=0):
		        fmt = '+'
		    ax2.plot(row['interplanar_angles'], row['rel_facet_size'],'o', label = row["legend"])
		ax2.set_xlabel('Angle (deg.)', fontsize = self.axes_fontsize)
		ax2.set_ylabel('Relative facet size', fontsize = self.axes_fontsize)
		ax2.grid(which='minor', alpha=0.2)
		ax2.grid(which='major', alpha=0.5)

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

		with open(prompt, 'rb') as f:
		    return pickle.load(f)