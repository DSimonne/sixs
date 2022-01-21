# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:44:13 2017

last modified:
Modified again the 09/11/2020

@author: andrea
"""

'''Small scripts to be used to load some of the binoculars outputs'''


#from libtiff import TIFF
### can also use imageio to save tiff####

#from vtk.util import numpy_support
#from tifffile3 import imsave

# plt.style.use('fivethirtyeight')

#directory= '/home/andrea/SciProj/MoS2/SXRD/22032017/binoculars_loc/'
#hdf5file1 = 'MoS2_hkl_Au_mask6_crop_296-310_[0.7-1.3, 0.7-1.3,-].hdf5'
#hdf5file2 = 'MoS2_hkl_Au_mask6_crop_77-92_[0.7-1.3, 0.7-1.3,-].hdf5'
#hdf5file = 'MoS2_hkl_Au_mask6_crop_254-294_[0.9-2.1, m1.1-1.1,-].hdf5'




import matplotlib.pyplot as plt
import pylab as pl
import os
from natsort import natsorted
import numpy as np
import tables
from tables import NoSuchNodeError
from phdutils.sixs import utilities3 as ut3
from matplotlib.colors import LogNorm
class hdf5load(object):
    '''Meant to return an object with the data matrix and the axes as attributes
    But better verify than sorry...
    It expects two strings:
        The folder name eg: "/home/user/data"
        the file name   eg: "Sample2_hkl_map_1.hdf5"  
    The object will contains two matrices: ratio between Counts and Contributions original matrices in the hdf5file
    the "data is manipulated to be consistet with the indexing [h, k, l], or [Qx, Qy, Qz].
    it therefore appear rotated compare to dataRaw."
    there are also the 3 (or 2) axes and the original vectors containinig the information to generate the axes.'''

    def __init__(self, directory, hdf5file):
        self.data = 0
        self.directory = directory
        self.fn = hdf5file
        fullpath = os.path.join(self.directory, self.fn)

        with tables.open_file(fullpath) as f:
            # works but utterly slow!!!!
            #ct = np.asanyarray( f.list_nodes('/binoculars/')[3],dtype = np.float32)
            #cont = np.asanyarray( f.list_nodes('/binoculars/')[2],dtype = np.float32)
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
        #f = bu3.hdf5load(hdf5dir, el)
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

    def hdf2tiff(self, axe, plot='YES', save='NO', axerange=None):
        '''Meant to plot/ save a hdf5 map from binoculars 
        axerange is a fraction of the range to be used insted of the full extension of it.'''
        #f = bu3.hdf5load(hdf5dir, el)
        if axerange == None:
            self.prjaxe(axe)
            img = self.img
        if axerange != None:
            self.prjaxe_range(axe, axerange)
            img = self.imgr
        if axe == 'H':
            axe1 = self.kaxe
            axe2 = self.laxe
            axe_name1 = 'K (rlu)'
            axe_name2 = 'L (rlu)'
        if axe == 'K':
            axe1 = self.haxe
            axe2 = self.laxe
            axe_name1 = 'H (rlu)'
            axe_name2 = 'L (rlu)'
        if axe == 'L':
            axe1 = self.haxe
            axe2 = self.kaxe
            axe_name1 = 'H (rlu)'
            axe_name2 = 'K (rlu)'

        if plot == 'YES':
            fig = plt.figure(num=None, figsize=(8, 6), dpi=80,
                             facecolor='w', edgecolor='k')
            # extent=[f.kaxe.min(),f.kaxe.max(),f.laxe.min(),f.laxe.max()]
            plt.imshow(img, cmap='jet', interpolation="nearest", origin="lower", aspect='auto',
                       norm=LogNorm(), vmin=0.01, vmax=2000, extent=[axe1.min(), axe1.max(), axe2.min(), axe2.max()])
            plt.title(self.fn[10:-5], fontsize=20)
            plt.xlabel(axe_name1, fontsize=20)
            plt.ylabel(axe_name2, fontsize=20)
            plt.tight_layout()
        if save == 'YES':
            tiff = TIFF.open(self.directory +
                             self.fn[0:-5] + '_' + axe + '.tiff', mode='w')
            tiff.write_image(self.img)
            tiff.close()
        return


def removeNan(x, y):
    '''it checks if any of the two vectors contains nan and remove them therein 
    and from its pair in the other vector'''
    if np.shape(y) == np.shape(x):
        ynan = np.argwhere(np.isnan(y))
        yf = np.delete(y, ynan)
        xf = np.delete(x, ynan)
        xnan = np.argwhere(np.isnan(xf))
        xff = np.delete(xf, xnan)
        yff = np.delete(yf, xnan)
    return xff, yff


# rod functions

def rodload(directory, rodFile):
    '''It reads the integreted rod file output of Binoculars-fitaid '''
    data = np.genfromtxt(directory + rodFile, dtype='float', delimiter='')
    return data


def txt2dat(directoryIN, filelist, rodlist, directoryOUT, fileout, LbraggList=[]):
    '''It takes 3 entry:
       the data directory,
       the filenames list of the Binoculars-fitaid integrated rods,
       rodlist is a list: [[h, k],[h, k]].... eg:[[-1, 1],[1, 0]]
       Lbragg parameter for the roughthness calculations, depends on the ctr
       The output directory
       the output file name. eg: filemane.dat
       It transform the data in a dat file that can be imported from ROD'''
    #print (np.shape(rodlist))
    #print (np.shape(filelist))
    if int(np.shape(rodlist)[0]) == int(np.shape(filelist)[0]):
        #print ('yes')
        g = open(directoryOUT+fileout, 'w+')
        g.write('Generated from Binoculars Files' + '\n')
        space = '    '
        if LbraggList == []:
            for n, el in enumerate(filelist):
                LbraggList.append(space)
        for n, el in enumerate(filelist):
            data = rodload(directoryIN, str(el))
            l, sf = removeNan(data[:, 0], data[:, 1])
            h = rodlist[n][0]
            k = rodlist[n][1]
            lb = LbraggList[n]
            for j in np.arange(np.shape(l)[0]):
                g.write(str(h) + space + str(k) + space + '%.3f' %
                        l[j] + space + '%.3f' % sf[j] + space + '%.3f' % (sf[j]*0.09) + space + '%.3f' % lb + '\n')
                #g.write(str(h) + space + str(k) + space + '\n')
                # print(l[j])
    return


def rodplot(directory, rodFile, color, scaleMax='NO', label=None, scFactor=1):
    '''It plots the integartions obtained/exported from binoculars-fitaid
    rodfile'''
    #plt.figure(num = None, figsize=(12, 7), dpi = 80, facecolor='w', edgecolor='k')
    data = np.genfromtxt(directory + rodFile, dtype='float',
                         delimiter='')  # , skip_header = 3)
    L0 = data[:, 0]
    Y0 = np.asanyarray(data[:, 1])
    L, Y = removeNan(L0, Y0)
    #L = L0
    #Y = Y0
    Int = Y * scFactor
    if scaleMax != 'NO':
        maxpos = np.nanargmax(Y)
        print((maxpos, Y[maxpos]))
        Int = Y/Y[maxpos]
    # data[:,1]=data[:,1]-min(data[:,1])
    if not label:
        #plt.plot(L, Int, color, linewidth = 2, label = rodFile[:-4])
        plt.plot(L, Int, color, marker='+', linewidth=0, label=rodFile[:-4])
    if (label != None and label != 'No'):
        #plt.plot(L, Int, color, linewidth = 2, label = label)
        plt.plot(L, Int, color, marker='+', linewidth=0, label=label)
    if label == 'No':
        #plt.plot(L, Int, color, linewidth = 2)
        plt.plot(L, Int, color, marker='+', linewidth=0)

    pl.semilogy()
    pl.xlim(min(L), max(L))
    pl.ylim(0, max(Y))
    if (label != None and label != 'No'):
        plt.legend(loc=1, fontsize=10)
    plt.grid(True)
    plt.show()


def rodplot_diff(directory, rodFile, bkgFile, color, plotOrig='NO'):
    '''It plots the integrations obtained/exported from binoculars-fitaid
    rodfile'''
    #plt.figure(num = None, figsize=(12, 7), dpi = 80, facecolor='w', edgecolor='k')
    data1 = rodload(directory, rodFile)
    data2 = rodload(directory, bkgFile)
    # data1 = np.genfromtxt(directory + rodFile, dtype= 'float', delimiter='')#, skip_header = 3)
    L1 = data1[:, 0]
    Y1 = np.nan_to_num(np.asanyarray(data1[:, 1]))
    #L1, Y1 = removeNan(L0, Y0)
    L2 = data2[:, 0]
    Y2 = np.nan_to_num(np.asanyarray(data2[:, 1]))
    #L2, Y2 = removeNan(L0, Y0)
    #L = L0
    #Y = Y0
    if plotOrig != 'NO':
        plt.plot(L1, Y1, 'k', marker='+', linewidth=0, label='Rod')
        plt.plot(L2, Y2, 'g', marker='+', linewidth=0, label='bkb')

    plt.plot(L1, Y1-Y2, color, marker='+', linewidth=0, label='Rod-bkg')

    pl.semilogy()
    pl.xlim(min(L1), max(L1))
    pl.ylim(0, max(Y1))
    plt.legend()
    plt.semilogy()
    plt.grid(True)
    plt.show()


def rodp(directory, rodFile, color, scaleMax='NO', label=None):
    '''Meant to plot the rods calculated by ROD.
    Not reallu a utility for binUtil but....'''
    data = np.genfromtxt(directory + rodFile, dtype='float',
                         delimiter='', skip_header=2)
    l = data[:, 2]
    Fsum = data[:, 6]
    if scaleMax != 'NO':
        maxpos = np.nanargmax(Fsum)
        print((maxpos, Fsum[maxpos]))
        Fsum = Fsum/Fsum[maxpos]
    # data[:,1]=data[:,1]-min(data[:,1])
    if label == None:
        #plt.plot(L, Int, color, linewidth = 2, label = rodFile[:-4])
        plt.plot(l, Fsum, color, marker='+', linewidth=0, label=rodFile[:-4])
    if (label != None and label != 'No'):
        #plt.plot(L, Int, color, linewidth = 2, label = label)
        plt.plot(l, Fsum, color, marker='+', linewidth=0, label=label)
    if label == 'No':
        #plt.plot(L, Int, color, linewidth = 2)
        plt.plot(l, Fsum, color, marker='+', linewidth=0)

    pl.semilogy()
    pl.xlim(min(l), max(l))
    #pl.xlim(0, 2.2)
    pl.ylim(0, max(Fsum))
    #pl.ylim(0.4, max(data[:,1]))

    pl.yticks(fontsize=18)
    pl.xticks(fontsize=18)
    pl.xlabel('L(rlu)', fontsize=20)
    pl.ylabel('sf (a.u.)', fontsize=20)
    #leg = plt.legend(loc = 'best', numpoints = 1, fancybox = True)
    # leg.get_frame().set_alpha(0.5)
    # plt.legend(bbox_to_anchor=(0., 1.04, 1., .102), loc = 3,
    #       ncol = 1, mode="expand", borderaxespad = 0.)
    plt.legend(loc=1, fontsize=14)
    plt.grid(True)
   #
    plt.show()
