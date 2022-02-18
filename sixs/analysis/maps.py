import matplotlib.pyplot as plt
import pylab as pl
import os
from natsort import natsorted
import numpy as np
import tables
from tables import NoSuchNodeError
from phdutils.sixs import utilities3 as ut3
from matplotlib.colors import LogNorm


class Map(object):
    """
    Meant to return an object with the data matrix and the axes as attributes
    It expects two strings:
        The folder name eg: "/home/user/data"
        the file name   eg: "Sample2_hkl_map_1.hdf5"  
    The object will contains two matrices: ratio between Counts and Contributions original matrices in the hdf5file
    the "data is manipulated to be consistet with the indexing [h, k, l], or [Qx, Qy, Qz].
    it therefore appear rotated compare to dataRaw."
    there are also the 3 (or 2) axes and the original vectors containinig the information to generate the axes.
    """

    def __init__(self, directory, hdf5file):
        self.data = 0
        self.directory = directory
        self.fn = hdf5file
        fullpath = os.path.join(self.directory, self.fn)

        with tables.open_file(fullpath) as f:
            ct = f.root.binoculars.counts.read()  # ~100 times faster
            cont = f.root.binoculars.contributions.read()
            self.dataRaw = np.divide(ct, cont, where=cont != 0)

            hkl = True
            QxQy = True
            QparQper = True
            Qphi = True
            Qindex = True

            try:
                H = f.list_nodes('/binoculars/')[0].H
                #K = f.list_nodes('/binoculars/')[0].K
                #L = f.list_nodes('/binoculars/')[0].L
            except NoSuchNodeError:
                hkl = False

            try:
                Qpar = f.list_nodes('/binoculars/')[0].Qpar
                #K = f.list_nodes('/binoculars/')[0].K
                #L = f.list_nodes('/binoculars/')[0].L
            except NoSuchNodeError:
                QparQper = False
            try:
                Index = f.list_nodes('/binoculars/')[0].Index
            except NoSuchNodeError:
                Qindex = False

            try:
                Index = f.list_nodes('/binoculars/')[0].Phi
                QxQy = False  # also Qphy can have Qz (or Qx, Qy)
            except NoSuchNodeError:
                Qphi = False

            if Qphi == False:  # also Qphy can have Qz (or Qx, Qy)
                try:
                    Qz = f.list_nodes('/binoculars/')[0].Qz
                #K = f.list_nodes('/binoculars/')[0].K
                #L = f.list_nodes('/binoculars/')[0].L
                except NoSuchNodeError:
                    QxQy = False

            if Qphi == True:
                self.data = self.dataRaw
                self.Phi = f.list_nodes('/binoculars/')[0].Phi[:]
                self.Q = f.list_nodes('/binoculars/')[0].Q[:]
                try:
                    self.Qxyz = f.list_nodes('/binoculars/')[0].Qx[:]
                except:
                    pass
                try:
                    self.Qxyz = f.list_nodes('/binoculars/')[0].Qy[:]
                except:
                    pass
                try:
                    self.Qxyz = f.list_nodes('/binoculars/')[0].Qz[:]
                except:
                    pass

            if Qindex == True:
                self.data = self.dataRaw
                self.index = f.list_nodes('/binoculars/')[0].Index[:]
                self.Q = f.list_nodes('/binoculars/')[0].Q[:]

            if hkl == True:
                self.data = np.swapaxes(self.dataRaw, 0, 2)
                self.H = f.list_nodes('/binoculars/')[0].H[:]
                self.K = f.list_nodes('/binoculars/')[0].K[:]
                self.L = f.list_nodes('/binoculars/')[0].L[:]

            if QxQy == True:

                self.data = self.dataRaw
                self.Z = f.list_nodes('/binoculars/')[0].Qz[:]
                self.X = f.list_nodes('/binoculars/')[0].Qx[:]
                self.Y = f.list_nodes('/binoculars/')[0].Qy[:]

            elif QparQper == True:
                self.data = self.dataRaw
                self.Y = f.list_nodes('/binoculars/')[0].Qper[:]
                self.X = f.list_nodes('/binoculars/')[0].Qpar[:]

            if Qphi == True:
                xaxe = np.linspace(self.Q[1], self.Q[2], 1+self.Q[5]-self.Q[4])
                self.Qaxe = xaxe
                yaxe = np.linspace(
                    self.Qxyz[1], self.Qxyz[2], 1+self.Qxyz[5]-self.Qxyz[4])
                self.Qxyzaxe = yaxe
                zaxe = np.linspace(
                    self.Phi[1], self.Phi[2], 1+self.Phi[5]-self.Phi[4])
                self.Phiaxe = zaxe

            if Qindex == True:
                self.qaxe = np.linspace(
                    self.Q[1], self.Q[2], 1+self.Q[5]-self.Q[4])
                self.indaxe = np.linspace(
                    self.index[1], self.index[2], 1+self.index[5]-self.index[4])

            if hkl == True:
                print("Data shape:", self.data.shape)
                xaxe = np.arange(self.H[1], self.H[2], 1+self.H[5]-self.H[4])
                self.haxe = np.linspace(
                    self.H[1], self.H[2], 1 + int(self.H[5] - self.H[4]))  # xaxe
                yaxe = np.arange(self.K[1], self.K[2], 1+self.K[5]-self.K[4])
                self.kaxe = np.linspace(
                    self.K[1], self.K[2], 1 + int(self.K[5] - self.K[4]))  # yaxe
                zaxe = np.arange(self.L[1], self.L[2], 1+self.L[5]-self.L[4])
                self.laxe = np.round(np.linspace(
                    self.L[1], self.L[2], 1 + int(self.L[5] - self.L[4])), 3)  # yaxe
                # self.laxe = np.round(np.linspace(self.L[1], self.L[2], int((self.L[2] - self.L[1])/self.L[3])), 3) #yaxe

            if QxQy == True:
                xaxe = np.linspace(
                    self.X[1], self.X[2], 1 + int(self.X[5]-self.X[4]))
                self.Qxaxe = xaxe
                yaxe = np.linspace(
                    self.Y[1], self.Y[2], 1 + int(self.Y[5]-self.Y[4]))
                self.Qyaxe = yaxe
                zaxe = np.linspace(
                    self.Z[1], self.Z[2], 1 + int(self.Z[5]-self.Z[4]))
                self.Qzaxe = zaxe

            if QparQper == True:
                xaxe = np.linspace(self.X[1], self.X[2], 1+self.X[5]-self.X[4])
                self.Qpar = xaxe
                yaxe = np.linspace(self.Y[1], self.Y[2], 1+self.Y[5]-self.Y[4])
                self.Qper = yaxe

            print("hkl,QxQy,QparQper,Qphi,Qindex",
                  hkl, QxQy, QparQper, Qphi, Qindex)

            return

    def prjaxe(self, axe):
        '''Project on one of the measured axes
        axe is a string: eg 'H'.
        the result is added as attribute .img to the file'''

        datanan = self.data
        if axe == 'H':
            axenum = 2
        if axe == 'K':
            axenum = 1
        if axe == 'L':
            axenum = 0

        if axe == 'Qx':
            # to be check good for projection along Qx
            datanan = np.swapaxes(self.dataRaw, 1, 2)
            axenum = 0

        if axe == 'Qy':
            # to be check good for projection along Qy
            datanan = np.swapaxes(self.dataRaw, 0, 2)
            axenum = 1

        if axe == 'Qz':
            # to be check good for projection along Qz
            datanan = np.swapaxes(self.dataRaw, 0, 1)
            axenum = 2

        self.img = np.nanmean(datanan, axis=axenum)
        return

    def prjaxe_range(self, axe, axerange):
        '''Project on one of the measured axes
        axe is a string: eg 'H'.
        the result is added as attribute .imgr to the file
        axerange is [0.8, 0.9] and define the positions of the value to be used
        in the array on the desired axe'''
        #datanan = self.data

        if axe == 'H':
            axenum = 2
            st = ut3.find_nearest(self.haxe, axerange[0])[0]
            nd = ut3.find_nearest(self.haxe, axerange[1])[0]
            datanan = self.data[:, :, st:nd]

        if axe == 'K':
            axenum = 1
            st = ut3.find_nearest(self.kaxe, axerange[0])[0]
            nd = ut3.find_nearest(self.kaxe, axerange[1])[0]
            datanan = self.data[:, st:nd, :]

        if axe == 'L':
            axenum = 0
            st = ut3.find_nearest(self.laxe, axerange[0])[0]
            nd = ut3.find_nearest(self.laxe, axerange[1])[0]
            datanan = self.data[st:nd, :, :]

        if axe == 'Qz':
            axenum = 2
            # to be check good for projection along Qz
            swap_data = np.swapaxes(self.dataRaw, 0, 1)

            st = ut3.find_nearest(self.Qzaxe, axerange[0])[0]
            nd = ut3.find_nearest(self.Qzaxe, axerange[1])[0]
            print(st, nd)
            datanan = swap_data[:, :, st:nd]

        if axe == 'Qy':
            axenum = 1
            # to be check good for projection along Qy
            swap_data = np.swapaxes(self.dataRaw, 0, 2)

            st = ut3.find_nearest(self.Qyaxe, axerange[0])[0]
            nd = ut3.find_nearest(self.Qyaxe, axerange[1])[0]
            print(st, nd)
            datanan = swap_data[:, st:nd, :]

        if axe == 'Qx':
            axenum = 0
            # to be check good for projection along Qx
            swap_data = np.swapaxes(self.dataRaw, 1, 2)

            st = ut3.find_nearest(self.Qxaxe, axerange[0])[0]
            nd = ut3.find_nearest(self.Qxaxe, axerange[1])[0]
            print(st, nd)
            datanan = swap_data[st:nd, :, :]

        self.imgr = np.nanmean(datanan, axis=axenum)
        return

    def prjaxes(self, axe1, axe2, range1=None, range2=None):
        '''Project on two of the measured axes
        axe is a string: eg 'H', 'L'.
        the result is added as attribute .int2 to the file'''
        datanan = self.data
        #axe1num = 10
        #axe2num = 10
        #axe3num = 10
        if axe1 == 'H':
            axe1num = 2
        if axe1 == 'K':
            axe1num = 1
        if axe1 == 'L':
            axe1num = 0
        if axe2 == 'H':
            axe2num = 2
        if axe2 == 'K':
            axe2num = 1
        if axe2 == 'L':
            axe2num = 0

        if axe2 == 'Phi':
            axe2num = 0
        if axe2 == 'Q':
            axe2num = 1
        if axe1 == 'Qxyz':
            axe1num = 2

        # if ax1 in ['L','Phi'] or ax1 in ['L','Phi']:
        #     axe1num = 0
        # if ax1 in ['K','Q'] or ax2 in ['K','Q'] :
        #     axe2num = 1
        # if ax1 in ['Qxyz','H'] or ax2 in ['Qxyz','H']:
        #     axe3num = 2

        if axe2num < axe1num:
            temp = np.nanmean(datanan, axis=axe1num)
            self.int2 = np.nanmean(temp, axis=axe2num)
        if axe2num > axe1num:
            temp = np.nanmean(datanan, axis=axe2num)
            self.int2 = np.nanmean(temp, axis=axe1num)
        return

    def hdf2png(self, axe, plot='YES', save='NO', axerange=None, vmax=2000, vmin=0.1, figsize=(16, 9), save_dir=False):
        '''Meant to plot/save a hdf5 map. 
        axerange is a fraction of the range to be used insted of the full extension of it.
        It can also be used only to plot if:
            arg: save = 'NO'  
        It can be used only to plot if:
            arg: plot = 'NO'   '''
        #f = Map(hdf5dir, el)
        if axerange == None:
            self.prjaxe(axe)
            img = self.img

        elif axerange != None:
            self.prjaxe_range(axe, axerange)
            img = self.imgr

        if axe == 'H':
            axe1 = self.kaxe
            axe2 = self.laxe
            axe_name1 = 'K (rlu)'
            axe_name2 = 'L (rlu)'

        elif axe == 'K':
            axe1 = self.haxe
            axe2 = self.laxe
            axe_name1 = 'H (rlu)'
            axe_name2 = 'L (rlu)'

        elif axe == 'L':
            axe1 = self.haxe
            axe2 = self.kaxe
            axe_name1 = 'H (rlu)'
            axe_name2 = 'K (rlu)'

        elif axe == 'Qxyz':
            axe1 = self.Qaxe
            axe2 = self.Phiaxe
            axe_name1 = 'Q'
            axe_name2 = 'Phi (deg)'

        elif axe == 'Qx':
            axe1 = self.Qyaxe
            axe2 = self.Qzaxe
            axe_name1 = 'Qy'
            axe_name2 = 'Qz'

        elif axe == 'Qy':
            axe1 = self.Qxaxe
            axe2 = self.Qzaxe
            axe_name1 = 'Qx'
            axe_name2 = 'Qz'

        elif axe == 'Qz':
            axe1 = self.Qxaxe
            axe2 = self.Qyaxe
            axe_name1 = 'Qx'
            axe_name2 = 'Qy'

        if plot == 'YES':
            plt.figure(figsize=figsize)

            # plt.text(1, 1, self.fn[-5], fontsize = 20)

            plt.imshow(img,
                       cmap='jet',
                       # interpolation="nearest",
                       origin="lower",
                       #aspect = 'auto',
                       norm=LogNorm(vmin=1e-2, vmax=1e1),
                       extent=[axe1.min(), axe1.max(), axe2.min(), axe2.max()]
                       )
            plt.xlabel(axe_name1, fontsize=30)
            plt.ylabel(axe_name2, fontsize=30)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            cbar = plt.colorbar(
                # orientation="horizontal",
                pad=0.1)

            cbar.ax.tick_params(labelsize=30)

            plt.tight_layout()

        if save == 'YES':
            if save_dir == False:
                save_dir = self.directory
            plt.savefig(save_dir+self.fn[:-5]+'_prj'+axe+'.png')
