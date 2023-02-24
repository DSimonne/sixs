#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:39:36 2019
Meant to open the data generated from the datarecorder upgrade of january 2019
Modified again and again... 
@author: andrea

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import tables  # h5py should take its place
import os
import numpy as np
import pickle
import time
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


# to help debug
from inspect import currentframe, getframeinfo
from inspect import currentframe


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno
##################################


class emptyO(object):
    '''Empty class used as container in the nxs2spec case. '''
    pass


class DataSet(object):
    '''Dataset read the file and store it in an object, from this object we can 
    retrive the data to use it.

         . Use as:
         dataObject = nxsRead4.DataSet( path/filename, path )
         filename can be '/path/filename.nxs' or just 'filename.nxs'
         directory is optional and it can contain '/dir00/dir01'
         both are meant to be string.

        . It returns an object with attibutes all the recorded sensor/motors 
        listed in the nxs file.
        . if the sensor/motor is not in the "aliad_dict" file it generate a 
        name from the device name.

    The object will also contains some basic methods for data reduction such as 
    ROI extraction, attenuation correction and plotting.
    Meant to be used on the data produced after the 11/03/2019 moment of a majour 
    upgrade of the datarecorder'''

    def __init__(self, filename, directory='', verbose='NO'):
        self.version = '16/12/2022'
        self.__version__ = self.version
        self.directory = directory
        self.filename = filename
        self.end_time = 2
        self.start_time = 1
        self.attlist = []
        self._list2D = []
        attlist = []  # used for self generated file attribute list
        aliases = []  # used for SBS imported with alias_dict file
        self. _coefPz = 1  # assigned just in case
        self. _coefPn = 1  # assigned just in case
        self.verbose = verbose  # explicits/prints some of the handled IO issues
        if directory == '':
            self.directory = '/'.join(filename.split('/')[:-1])+'/'

        if self.verbose != 'NO':
            print('#######################################################')
            print('Treating', filename)
        try:
            path = os.path.dirname(__file__)
            pathAlias = os.path.join(path, 'alias_dict.txt')
            self._alias_dict = pickle.load(open(pathAlias, 'rb'))
            #
        except:  # FileNotFoundError:
            # print('First Alias Not Found')
            try:
                self._alias_dict = pickle.load(
                    open('/home/experiences/sixs/com-sixs/python/alias_dict.txt', 'rb'))
            except:
                print('NO ALIAS FILE')
                self._alias_dict = None

        def is_empty(any_structure):
            '''Quick function to determine if an array, tuple or string is 
            empty '''
            if any_structure:
                return False
            else:
                return True
        # Load the file
        fullpath = os.path.join(self.directory, self.filename)
        try:
            print(fullpath)
            ff = tables.open_file(fullpath, 'r')
            f = ff.list_nodes('/')[0]

        except:
            if self.verbose != 'NO':
                print('Tables error reading/opening the file',
                      ' Line: ', get_linenumber())
                # raise
            return
        # check if any scanned data a are present
        try:
            if not hasattr(f, 'scan_data'):
                self.scantype = 'unknown'
                ff.close()
                return
        except:
            if self.verbose != 'NO':
                print('Scantype Unknown')
            return

        # Discriminating between SBS or FLY scans

        try:
            if f.scan_data.data_01.name == 'data_01':
                self.scantype = 'SBS'
        except tables.NoSuchNodeError:
            self.scantype = 'FLY'

        ########################## Reading FLY ################################
        if self.scantype == 'FLY':
            # generating the attributes with the recorded scanned data
            dataL = 0
            for leaf in f.scan_data:
                list.append(attlist, leaf.name)
                self.__dict__[leaf.name] = leaf[:]
                # selecting the longest data vector which it should be many of the same lenght
                if dataL < len(leaf[:]):
                    dataL = len(leaf[:])
                time.sleep(0.1)
            self.attlist = attlist
            try:  # adding just in case eventual missing attenuation
                if not hasattr(self, 'attenuation'):
                    self.NoAttenuation = 'verify data file: Attenuation'
                    self.attlist.append('NoAttenuation')
                    # np.zeros(dataL) #removed to avoid to generate unreal data
                    self.attenuation = 'NoAttenuation'
                    # self.attlist.append('attenuation') #removed to avoid to generate unreal data
                    if self.verbose != 'NO':
                        print('verify data file: Attenuation',
                              ' Line: ', get_linenumber())
            except:
                pass
            try:
                if not hasattr(self, 'attenuation_old'):
                    self.NoAttenuation_old = 'verify data file: Attenuation old'
                    self.attlist.append('NoAttenuation_old')
                    self.attenuation_old = 'NoAttenuation_old'
                    # self.attlist.append('attenuation_old')
                    if self.verbose != 'NO':
                        print('verify data file: Attenuation old',
                              ' Line: ', get_linenumber())
            except:
                pass

       ###################### Reading SBS ####################################
        if self.scantype == 'SBS':
            dataL = 0
            if self._alias_dict:  # Reading with dictionary
                for leaf in f.scan_data:
                    try:
                        alias = self._alias_dict[leaf.attrs.long_name.decode(
                            'UTF-8')]
                        if alias not in aliases:
                            aliases.append(alias)
                            self.__dict__[alias] = leaf[:]
                            # selecting the longest data vector which it should be many of the same lenght
                            if dataL < len(leaf[:]):
                                dataL = len(leaf[:])

                    except:
                        self.__dict__[
                            leaf.attrs.long_name.decode('UTF-8')] = leaf[:]
                        aliases.append(leaf.attrs.long_name.decode('UTF-8'))
                        pass
                self.attlist = aliases
                if hasattr(self, 'sensorsTimestamps'):
                    self.epoch = self.__getattribute__('sensorsTimestamps')
                    list.append(self.attlist, 'epoch')

            else:
                for leaf in f.scan_data:  # Reading with dictionary
                    # generating the attributes with the recorded scanned data
                    attr = leaf.attrs.long_name.decode('UTF-8')
                    attrshort = leaf.attrs.long_name.decode(
                        'UTF-8').split('/')[-1]
                    attrlong = leaf.attrs.long_name.decode(
                        'UTF-8').split('/')[-2:]
                    if attrshort not in attlist:
                        # rename the sensortimestamps as epoch
                        if attr.split('/')[-1] == 'sensorsTimestamps':
                            list.append(attlist, 'epoch')
                            self.__dict__['epoch'] = leaf[:]
                        else:
                            list.append(attlist, attr.split('/')[-1])
                            self.__dict__[attr.split('/')[-1]] = leaf[:]
                    else:  # Dealing with for double naming
                        list.append(attlist, '_'.join(attrlong))
                        self.__dict__['_'.join(attrlong)] = leaf[:]
                self.attlist = attlist

            try:  # adding just in case eventual missing attenuation
                if not hasattr(self, 'attenuation_old'):
                    # new att_old name? 15/12/2022
                    self.attenuation_old = self.att_old_sbs_xpad[:]
                    # Att old in SBS are 0-63 for Al and >64 for Ag
                    if (not isinstance(self.attenuation_old, str)):
                        if max(self.attenuation_old > 63):
                            self.attenuation_old = self.attenuation_old/64  # filter old silver set
                    list.append(self.attlist, 'attenuation_old')
            except:
                print('No attenuations SBS')
                self.attenuation = 'NoAttenuation old'

            try:  # adding just in case eventual missing attenuation
                if not hasattr(self, 'attenuation'):
                    self.attenuation = 'NoAttenuation'  # later will be filtered between str or values
                    list.append(self.attlist, 'attenuation')
            except:
                print('No attenuations SBS')
                self.attenuation = 'NoAttenuation'

            try:  # adding just in case eventual missing attenuation
                if not hasattr(self, 'attenuation') and not hasattr(self, 'attenuation_old'):
                    print('verify data file: Attenuation, line: ',
                          get_linenumber())
                    self.NoAttenuation = 'verify data file: Attenuation'
                    self.attlist.append('NoAttenuation')
                    self.attenuation = 'NoAttenuation'
            except:
                if self.verbose != 'NO':
                    print('verify data file line: ', get_linenumber())

        ##################################################################################################################################
        # patch xpad140 / xpad70
        # try to correct wrong/inconsistency naming coming from FLY/SBS name system
        # It is fragile as idea but...
        # BL2D = {120:'xpad70',240:'xpad140',515:'merlin',512:'maxipix', 1065:'eiger',1040:'cam2',256:'ufxc'}
        BL2D = {120: 'xpad70', 240: 'xpad140', 512: 'merlin',
                1065: 'eiger', 1040: 'cam2', 256: 'ufxc'}
        self._BL2D = BL2D

        try:
            list2D = self.det2d()  # generating the list self._list2D
            # print(list2D)
            for el in list2D:
                detarray = self.getStack(el)
                detsize = detarray.shape[1]  # the key for the dictionary
                # print(detsize)
                if detsize in BL2D:
                    detname = BL2D[detsize]  # the detector name from the size
                    if not hasattr(self, detname):
                        # print('adding:', detname)
                        # adding the new attrinute name
                        self.__setattr__(detname, detarray)
                        self.attlist.append(detname)
                        # removing the wrong/incosistent naming due to SBS/FLY naming.
                        self.__delattr__(el)
                        # removing the wrong/incosistent naming due to SBS/FLY naming.
                        self.attlist.remove(el)
                        # print('removed:', el)
                    # if hasattr(self, detname):
                    #    print('detector already detected')
                if detsize not in BL2D:
                    if self.verbose != 'NO':
                        print('Detected a not standard detector: check ReadNxs4')

            list2D = self.det2d()  # re-generating the list self._list2D
            # for el in list2D:
            #     if 'image' in el: #remove the " detector'_image' " name and its matrix once the attribute 'detector is created'
            #         list2D.remove(el)
            #         self.__delattr__(el)
            #         self.attlist.remove(el)
            self._list2D = list2D
        except:
            if self.verbose != 'NO':
                print('2D issue')

         ### adding some useful attributes common between SBS and FLY#########################################################
        try:
            # apparently in some occasion we can be without mono :-/
            mono = f.SIXS.__getattr__('i14-c-c02-op-mono')
            self.waveL = mono.__getattr__('lambda')[0]
            self.attlist.append('waveL')
            self.energymono = mono.energy[0]
            self.attlist.append('energymono')
        except (tables.NoSuchNodeError):
            pass
        try:  # this is obsolete but...
            # apparently in some occasion we can be without mono  :-/
            self.energymono = f.SIXS.Monochromator.energy[0]
            self.attlist.append('energymono')
            self.waveL = f.SIXS.Monochromator.wavelength[0]
            self.attlist.append('waveL')
        except:
            pass
        # probing time stamps and eventually use epoch to rebuild them
        if hasattr(f, 'start_time'):
            try:
                st = f.start_time[...]
                self.start_time = time.mktime(
                    time.strptime(st.decode(), '%Y-%m-%dT%H:%M:%S'))
            except:
                if is_empty(np.shape(f.start_time)):
                    try:
                        self.start_time = self.epoch.min()
                    except AttributeError:
                        self.start_time = 1605740001  # impose time for a post 2020 treatment
                        if self.verbose != 'NO':
                            print('File has time stamps issues',
                                  ' Line: ', get_linenumber())
                else:
                    self.start_time = f.start_time[0]
        # sometimes this attribute is absent, especially on the ctrl+C scans
        elif not hasattr(f, 'start_time'):
            if self.verbose != 'NO':
                print('File has time stamps issues')
            self.start_time = 1605740001  # necessary for nxs2spec conversion
        if hasattr(f, 'end_time'):  # sometimes this attribute is absent, especially on the ctrl+C scans
            try:
                nd = f.end_time[...]
                self.end_time = time.mktime(
                    time.strptime(nd.decode(), '%Y-%m-%dT%H:%M:%S'))
            except:
                if is_empty(np.shape(f.end_time)):
                    try:
                        self.end_time = self.epoch.max()
                    except AttributeError:
                        self.end_time = self.start_time + 2
                        if self.verbose != 'NO':
                            print('File has time stamps issues',
                                  ' Line: ', get_linenumber())
                else:
                    self.end_time = f.end_time[0]
        elif not hasattr(f, 'end_time'):
            if self.verbose != 'NO':
                print('File has time stamps issues l304')
            self.end_time = self.start_time + 2  # necessary for nxs2spec conversion

        try:  # att_coef
            self._coefPz = f.SIXS._f_get_child(
                'i14-c-c00-ex-config-att').att_coef[0]  # coeff piezo
            self.attlist.append('_coefPz')
        except:
            if self.verbose != 'NO':
                print('No att coef new', ' Line: ', get_linenumber())
        try:
            self._coefPn = f.SIXS._f_get_child(
                'i14-c-c00-ex-config-att-old').att_coef[0]  # coeff Pneumatic
            self.attlist.append('_coefPn')
        except:
            if self.verbose != 'NO':
                print('No att coef old', ' Line: ', get_linenumber())

        try:  # sometimes the publisher loose memory of the ROIs, use the roicounters instead
            GroupList = f.SIXS._g_list_group(f.SIXS)
            for el in GroupList[0]:
                if 'roicounter' in el:
                    # print(el)
                    tmp1 = f.SIXS._f_get_child(el)
                    x = []
                    y = []
                    w = []
                    h = []
                    # looking at __height, __width, __x, __y (should be in w=this order)
                    for leaf in tmp1:
                        for el1 in leaf:
                            datastr = (el1.tobytes().decode('utf8'))
                            # data.append([int(i) for i in (datastr.split('\n'))])
                            if 'height' in leaf.name:
                                h = np.array([int(i)
                                             for i in (datastr.split('\n'))])
                            if 'width' in leaf.name:
                                w = np.array([int(i)
                                             for i in (datastr.split('\n'))])
                            if '__x' in leaf.name:
                                x = np.array([int(i)
                                             for i in (datastr.split('\n'))])
                            if '__y' in leaf.name:
                                y = np.array([int(i)
                                             for i in (datastr.split('\n'))])

                    # to be sure that are stored in the right order!
                    data = np.c_[x, y, w, h]
                    if el == 'i14-c-c00-dt-xpad.s140-roicounters':
                        self.__dict__['_roi_limits_xpad140'] = data
                        list.append(self.attlist, '_roi_limits_xpad140')
                    if el == 'i14-c-c00-dt-xpad.s70-roicounters':         # 'i14-c-c00/dt/xpad.s70-roicounters'
                        self.__dict__['_roi_limits_xpad70'] = data
                        list.append(self.attlist, '_roi_limits_xpad70')
                    # 'i14-c-c00/dt/merlin-quad-roicounters':
                    if 'merlin' in el:
                        self.__dict__['_roi_limits_merlin'] = data
                        list.append(self.attlist, '_roi_limits_merlin')
                    if el == 'i14-c-c00-dt-eiger.1-roicounters':         # 'i14-c-c00/dt/eiger.1-roicounters'
                        self.__dict__['_roi_limits_eiger'] = data
                        list.append(self.attlist, '_roi_limits_eiger')
                    if el == 'i14-c-c00-dt-maxipix.det-roicounters':     # 'i14-c-c00/dt/maxipix.det-roicounters'
                        self.__dict__['_roi_limits_maxipix'] = data
                        list.append(self.attlist, '_roi_limits_maxipix')

        except:
            if self.verbose != 'NO':
                raise
                print('RoiCounter(s) issue  ', el, ' Line: ', get_linenumber())

            ######################################### xpad/ 2D ROIs   ###########################################

        try:  # kept for "files before 18/11/2020   related to xpad70/140 transition"
            if self.start_time < 1605740000:                   # apply to file older than Wed Nov 18 23:53:20 2020
                # print('old file')
                self. _roi_limits_xpad140 = f.SIXS._f_get_child(
                    'i14-c-c00-ex-config-publisher').roi_limits[:][:]
                self.attlist.append('_roi_limits_xpad140')
                # self.roi_names = str(f.SIXS._f_get_child('i14-c-c00-ex-config-publisher').roi_name.read()).split()
                roi_names_cell = f.SIXS._f_get_child(
                    'i14-c-c00-ex-config-publisher').roi_name.read()
                self._roi_names_xpad140 = roi_names_cell.tolist().decode().split('\n')
                self.attlist.append('_roi_names_xpad140')
                self._ifmask_xpad140 = f.SIXS._f_get_child(
                    'i14-c-c00-ex-config-publisher').ifmask[:]
                self.attlist.append('_ifmask_xpad140')
                try:
                    self._mask_xpad140 = f.SIXS._f_get_child(
                        'i14-c-c00-ex-config-publisher').mask[:]
                    self.attlist.append('_mask_xpad140')
                except:
                    if self.verbose != 'NO':
                        print('No Mask', ' Line: ', get_linenumber())
            if self.start_time > 1605740000:                    # apply to file after  Wed Nov 18 23:53:20 2020
                dets = self._list2D  # the 2D detector list potentially extend here for the eiger ROIs
                for el in dets:
                    if el == 'xpad70':
                        # print('set xpad70')
                        # self._roi_limits_xpad70 = f.SIXS._f_get_child('i14-c-c00-ex-config-xpad70').roi_limits[:][:]
                        # print('ok limits', self._roi_limits_xpad70)
                        # self.attlist.append('_roi_limits_xpad70')
                        self._distance_xpad70 = f.SIXS._f_get_child(
                            'i14-c-c00-ex-config-xpads70').distance_xpad[:]  # mind the "s"
                        # print('ok distance', self._distance_xpad70)
                        self.attlist.append('_distance_xpad70')
                        self._ifmask_xpad70 = f.SIXS._f_get_child(
                            'i14-c-c00-ex-config-xpads70').ifmask[:]  # mind the "s"
                        # print('ok ifmask', self._ifmask_xpad70)
                        self.attlist.append('_ifmask_xpad70')
                        try:
                            self._mask_xpad70 = f.SIXS._f_get_child(
                                'i14-c-c00-ex-config-xpads70').mask[:]  # mind the "s"
                            self.attlist.append('_mask_xpad70')
                        except:
                            if self.verbose != 'NO':
                                print('no mask xpad70',
                                      ' Line: ', get_linenumber())
                        # roi_names_cell = f.SIXS._f_get_child('i14-c-c00-ex-config-xpad70').roi_name.read()
                        # self._roi_names_xpad70 = roi_names_cell.tolist().decode().split('\n')
                        # self.attlist.append('_roi_names_xpad70')

                    if el == 'xpad140':
                        # print('xpad140')
                        # self._roi_limits_xpad140 = f.SIXS._f_get_child('i14-c-c00-ex-config-xpad140').roi_limits[:][:]
                        # self.attlist.append('_roi_limits_xpad140')
                        self._distance_xpad140 = f.SIXS._f_get_child(
                            'i14-c-c00-ex-config-xpads140').distance_xpad[:]  # mind the "s"
                        self.attlist.append('_distance_xpad140')
                        self._ifmask_xpad140 = f.SIXS._f_get_child(
                            'i14-c-c00-ex-config-xpads140').ifmask[0]  # mind the "s"
                        self.attlist.append('_ifmask_xpad140')
                        try:
                            # print('check mask l 360')
                            self._mask_xpad140 = f.SIXS._f_get_child(
                                'i14-c-c00-ex-config-xpads140').mask[:]  # mind the "s"
                            self.attlist.append('_mask_xpad140')
                        except:
                            if self.verbose != 'NO':
                                print('no mask xpad140',
                                      ' Line: ', get_linenumber())
                        # roi_names_cell = f.SIXS._f_get_child('i14-c-c00-ex-config-xpad140').roi_name.read()
                        # self._roi_names_xpad140 = roi_names_cell.tolist().decode().split('\n')
                        # self.attlist.append('_roi_names_xpad140')
                    if el == 'merlin':
                        # print('xpad140')
                        # self._roi_limits_merlin = f.SIXS._f_get_child('i14-c-c00-ex-config-merlin').roi_limits[:][:]
                        # self.attlist.append('_roi_limits_merlin')
                        self._distance_merlin = f.SIXS._f_get_child(
                            'i14-c-c00-ex-config-merlin').distance_xpad[:]
                        self.attlist.append('_distance_merlin')
                        self._ifmask_merlin = f.SIXS._f_get_child(
                            'i14-c-c00-ex-config-merlin').ifmask[:]
                        self.attlist.append('_ifmask_merlin')
                        try:
                            # print('check mask l 360')
                            self._mask_merlin = f.SIXS._f_get_child(
                                'i14-c-c00-ex-config-merlin').mask[:]
                            self.attlist.append('_mask_merlin')
                        except:
                            if self.verbose != 'NO':
                                print('no mask merlin',
                                      ' Line: ', get_linenumber())
                        # roi_names_cell = f.SIXS._f_get_child('i14-c-c00-ex-config-merlin').roi_name.read()
                        # self._roi_names_merlin = roi_names_cell.tolist().decode().split('\n')
                        # self.attlist.append('_roi_names_merlin')

        except:
            if self.verbose != 'NO':
                print('Issues with Publisher(s)', el,
                      ' Line: ', get_linenumber())
                # raise #Exception
##############################
        # 08/06/2022 the saved integration time from the timescan is wrong and influenced by the former "ct" command
        # so this part is commented for now

        # try:
        #     ### Note for the future, maybe I should look for "c00-dt-" should work for any lima-device
        #     for leaf in f.SIXS:
        #         #print(leaf._v_name)
        #         if 'c00-dt-' in leaf._v_name :
        #             #print(leaf._v_name)
        #             for el in self._list2D: #among the 2D detectorse saved (all of them should have the same time up to now)
        #                 if el[-2:] in leaf._v_name: # in the file is saved as xpad.s140 so... blooody point force to use shorter strings
        #                     if 'roicounters' not in leaf._v_name:
        #                         try:
        #                             self._integration_time = f.SIXS._f_get_child(leaf._v_name).exposure_time[0]/1000
        #                             self.attlist.append('_integration_time')
        #                             if self.verbose!='NO':
        #                                 print('OK Int Time:', self._integration_time)
        #                         except:
        #                             if self.verbose!='NO':
        #                                 print('Warning NO Int Time:')

#######################################
  #          /com/SIXS/i14-c-c00-ex-config-global
        try:
            self._integration_time = f.SIXS._f_get_child(
                'i14-c-c00-ex-config-global').integration_time[0]
            self.attlist.append('_integration_time')
        except:
            # /com/SIXS/i14-c-c00-ex-config-global SBS save the integration time in the data also
            if hasattr(self, 'integration_time'):
                self._integration_time = self.integration_time[0]
            else:
                self._integration_time = 'NoIntagrationTime'
            # self.attlist.append('_coef')
                if self.verbose != 'NO':
                    print('No integration time defined',
                          ' Line: ', get_linenumber())

        ff.close()

    ########################################################################################
    ##################### down here useful function in the NxsRead #########################
    def getStack(self, Det2D_name):
        '''For a given  2D detector name given as string it check in the 
        attribute-list and return a stack of images'''
        try:
            stack = self.__getattribute__(Det2D_name)
            return stack
        except:
            if self.verbose != 'NO':
                print('There is no such attribute',
                      ' Line: ', get_linenumber())

    def make_maskFrame_xpad(self):
        '''It generate a new attribute 'mask0_xpad' to remove the double pixels
        it can be applied only to xpad140 for now.'''
#    f = tables.open_file(filename)
#    scan_data = f.list_nodes('/')[0].scan_data
        detlist = self.det2d()

        if 'xpad140' in detlist:
            mask = np.zeros((240, 560), dtype='bool')
            mask[:, 79:560:80] = True
            mask[:, 80:561:80] = True
            mask[119, :] = True
            mask[120, :] = True
            mask[:, 559] = False
            self.mask0_xpad140 = mask
            self.attlist.append('mask0_xpad140')
        if 'xpad70' in detlist:
            mask = np.zeros((120, 560), dtype='bool')
            mask[:, 79:559:80] = True
            mask[:, 80:560:80] = True
            self.mask0_xpad70 = mask
            self.attlist.append('mask0_xpad70')
        return

    def roi_sum(self, stack, roi):
        '''given a stack of images it returns the integals over the ROI  
        roi is expected as eg: [257, 126,  40,  40] '''
        return stack[:, roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]].sum(axis=1).sum(axis=1).copy()

    def roi_sum_mask(self, stack, roi, mask):
        '''given a stack of images it returns the integals over the ROI minus 
        the masked pixels  
        the ROI is expected as eg: [257, 126,  40,  40] '''
        _stack = stack[:]*(1-mask.astype('uint16'))
        return _stack[:, roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]].sum(axis=1).sum(axis=1)

    def calcFattenuation(self, attcoef='default', filters='default'):
        '''It aims to calculate: (attcoef**filters[:]))/acqTime considering the presence of att_new and old '''
        # if _coefPz and _coefPn are not string and .attenuation is not a string... apply otherwise...
        # generate a string "NO_fattenuation"
        if attcoef == 'default' and filters == 'default':
            if (not isinstance(self.attenuation, str)) and (not isinstance(self.attenuation_old, str)):  # both valid values
                self._fattenuations = (
                    self._coefPz**self.attenuation)*(self._coefPn**self.attenuation_old)

            # Piezo==> values,  old==> string
            if (not isinstance(self.attenuation, str) and isinstance(self.attenuation_old, str)):
                # use only the piezo and forget the other
                self._fattenuations = (self._coefPz**self.attenuation)

            # Piezo==> string,  old==> value
            if (isinstance(self.attenuation, str) and (not isinstance(self.attenuation_old, str))):
                # use only the old and forget the other
                self._fattenuations = (self._coefPn**self.attenuation_old)

            # none of the filterssets are collected correctly
            if isinstance(self.attenuation, str) and isinstance(self.attenuation_old, str):
                self._fattenuations = 'NO_fattenuation_filters_issue'

        # the minimum of the other cases, always refer to the Pz filters
        if attcoef != 'default' and filters == 'default':
            if not (isinstance(self.attenuation, str) and isinstance(self.attenuation_old, str)):  # both valid values
                # new coeff appled only to piezo
                self._fattenuations = (
                    attcoef**self.attenuation)*(self._coefPn**self.attenuation_old)

            # Piezo==> values,  old==> string
            if not (isinstance(self.attenuation, str) and not isinstance(self.attenuation_old, str)):
                # use only the piezo and forget the other
                self._fattenuations = (attcoef**self.attenuation)

            if isinstance(self.attenuation, str):
                # NO attenuation ==> No _fattenuation
                self._fattenuations = 'NO_fattenuation_filters_issue'
            # self._fattenuations = attcoef**self.attenuation

        if attcoef == 'default' and filters != 'default':  # use the piezo att on the provided filters
            self._fattenuations = self.coefPz**filters[:]

        if attcoef != 'default' and filters != 'default':  # using the provided settings
            self._fattenuations = attcoef**filters[:]

    def numIntROI(self, stack, roiextent, maskname, ROIname):
        '''Just the numerical integral of the ROI only corrected by the mask
        roiextent is expected as eg: [257, 126,  40,  40]'''

        if hasattr(self, maskname):
            mask = self.__getattribute__(maskname)
            roiM = self.roi_sum_mask(stack, roiextent, mask)
            setattr(self, ROIname, roiM)
            if ROIname not in self.attlist:
                self.attlist.append(ROIname)

        if not hasattr(self, maskname):  # verify that thereis NO mask
            # print(maskname)
            roiM = self.roi_sum(stack, roiextent)
            setattr(self, ROIname, roiM)
            # integrals = self.roi_sum( stack,roiextent)
            if self.verbose != 'NO':
                print('NO Mask ==> Brutal Numerical Integration Only.',
                      ' Line: ', get_linenumber())

        return  # NO Mask ==> No correction!

    def calcROI(self, stack, roiextent, maskname, acqTime, ROIname, coef='default', filt='default'):
        '''To calculate the roi corrected by attcoef, mask, filters, 
        acquisition_time ROIname is the name of the attribute that will be attached to the dataset object
        mind that there might be a shift between motors and filters in the SBS scans
        the ROI is expected as eg: [257, 126,  40,  40] '''
        # here is generated the self._fattenuations: either as string either as vector
        self.calcFattenuation(attcoef=coef, filters=filt)
        if hasattr(self, maskname):
            mask = self.__getattribute__(maskname)
            integrals = self.roi_sum_mask(stack, roiextent, mask)
        if not hasattr(self, maskname):  # verify that thereis NO mask

            # if No mask still correct for filters and time... see calcROI_auto3
            fakemask = np.zeros(np.shape(stack[0]))
            # integrals = 'No Integrals'
            # print(get_linenumber())
            integrals = self.roi_sum_mask(stack, roiextent, fakemask)
            # integrals = self.roi_sum( stack,roiextent)
            if self.verbose != 'NO':
                print('NO Mask ==> No Mask correction',
                      ' Line: ', get_linenumber())
            # return # NO Mask ==> No correction!

        if self.scantype == 'SBS':   # here handling the data shift between data and filters SBS
            # NO fattenuation ==> No correction
            if not isinstance(self._fattenuations, str):
                _filterchanges = np.where(
                    (self._fattenuations[1:]-self._fattenuations[:-1]) != 0)

            # print(self._fattenuations)
                # print(acqTime)
                # integrals must exist as vector, as well as _filterchanges
                if (not isinstance(integrals, str)) and (not isinstance(acqTime, str)):
                    roiC = (integrals[:]*self._fattenuations)/acqTime
                    _filterchanges = np.asanyarray(_filterchanges)
                    np.put(roiC, _filterchanges+1, np.NaN)

                    setattr(self, ROIname, roiC)
                    if ROIname not in self.attlist:
                        self.attlist.append(ROIname)
                if (isinstance(integrals, str)) or (isinstance(acqTime, str)):
                    if self.verbose != 'NO':
                        print('Missing Something for ROI corrections ==> No correction',
                              ' Line: ', get_linenumber())

        if self.scantype == 'FLY':
            if not isinstance(self._fattenuations, str):
                # here handling the data shift between data and filters FLY
                f_shi = np.concatenate(
                    (self._fattenuations[2:], self._fattenuations[-1:], self._fattenuations[-1:]))
                # integrals must exist as vector, as well as _filterchanges
                if (not isinstance(integrals, str)) and (not isinstance(acqTime, str)):
                    roiC = (integrals[:]*(f_shi[:]))/acqTime
            # print(acqTime)
                    setattr(self, ROIname, roiC)
                    if ROIname not in self.attlist:
                        self.attlist.append(ROIname)
                if (isinstance(integrals, str)) or (isinstance(acqTime, str)):
                    if self.verbose != 'NO':
                        print('Missing Something for ROI corrections ==> No correction',
                              ' Line: ', get_linenumber())
        return

    def plotRoi(self, motor, roi, color='-og', detname=None, Label=None, mask='No'):
        '''It integrates the desired roi and plot it
        this plot function is simply meant as quick verification.
            Motor: motor name string
            roi: is the roi name string of the desired region measured or in the form :[257, 126,  40,  40]
            detname: detector name;  it used first detector it finds if not differently specified  '''
        if not detname:
            detname = self.det2d()[0]
            print(detname)

        if motor in self.attlist:
            xmot = getattr(self, motor)
        if detname:
            # detname = self.det2d()[0]
            stack = self.getStack(detname)
            if isinstance(roi, str):
                roiArr = self._roi_limits[self._roi_names.index(roi)]
            if isinstance(roi, list):
                roiArr = roi
            yint = self.roi_sum(stack, roiArr)
        # print(np.shape(xmot), np.shape(yint))
        plt.plot(xmot, yint, color, label=Label)

    def plotscan(self, Xvar, Yvar, color='-og', Label=None, mask='No'):
        '''It plots Xvar vs Yvar.
        Xvar and Yvar must be in the attributes list'''
        if Xvar in self.attlist:
            x = getattr(self, Xvar)
            print('x ok')
        if Yvar in self.attlist:
            y = getattr(self, Yvar)
            print('y ok')
        plt.plot(x, y, color, label=Label)

    def calcROI_auto3(self):
        '''if exist _coef, _integration_time, _roi_limits, _roi_names it can be applied
        to recalculate the roi on one or more 2D detectors.
        it configure the se of calcROI for multiple detectors/mask 
        filters and motors are shifted of one points for the FLY. corrected in the self.calcROI
        For SBS the data point when the filter is changed is collected with no constant absorber and therefore is rejected.
        replaced by the calcROI_auto2 '''
        # calcROI(self, stack,roiextent, maskname,attcoef, filters, acqTime, ROIname):
        if self.det2d():
            if self.scantype == 'SBS':  # SBS Correction #######################################
                if hasattr(self, '_npts'):
                    print('Auto ROI already applied')
                # check if the process was alredy runned once on this object
                if not hasattr(self, '_npts'):
                    self._npts = len(self.__getattribute__(self._list2D[0]))
                    for el in self._list2D:
                        try:
                            stack = self.__getattribute__(el)
                            if self.__getattribute__('_ifmask_'+el) and hasattr(self, '_mask_' + el):
                                maskname = '_mask_' + el
                                # print(maskname)
                            # if not self.__getattribute__('_ifmask_'+el)or not hasattr(self,'_mask_' + el):
                            if not hasattr(self, '_mask_' + el):  # disregard the _ifmask
                                maskname = 'No_Mask'  # not existent attribute filtered away from the roi_sum function
                            for pos, roi in enumerate(self.__getattribute__('_roi_limits_' + el), start=0):
                                if maskname != 'No_Mask' and self.__getattribute__('_mask_' + el) != 'No_Mask':
                                    # roiname = self.__getattribute__('_roi_names_' + el)[pos] +'_'+el+'_new'
                                    # Mask Filters Time
                                    roiname1 = 'roi'+str(pos) + '_'+el+'_MFT'
                                    # print(roiname)
                                    self.calcROI(
                                        stack, roi, maskname, self._integration_time, roiname1, coef='default', filt='default')
                                    # Mask only correction
                                    roiname2 = 'roi'+str(pos) + '_'+el+'_M'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)
                                if maskname == 'No_Mask' or self.__getattribute__('_mask_' + el) == 'No_Mask':
                                    # Mask only correction
                                    roiname2 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)
                                    if roiname2 not in self.attlist:
                                        self.attlist.append(roiname2)
                                        # workaround the absence of mask but still correcting the filters
                                    # Mask only correction
                                    roiname3 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly_FT'
                                    self.calcROI(
                                        stack, roi, maskname, roiname3)
                                    if roiname3 not in self.attlist:
                                        self.attlist.append(roiname3)

                        except:
                            # raise
                            if self.verbose != 'NO':
                                print(
                                    'issues with ', el, 'data Matrix/mask/roi_limit', ' Line: ', get_linenumber())

            if self.scantype == 'FLY':  # FLY correction ##################################

                if hasattr(self, '_npts'):
                    print('Auto ROI already applied')
                # check if the process was alredy runned once on this object
                if not hasattr(self, '_npts'):
                    self._npts = len(self.__getattribute__(self._list2D[0]))
                    for el in self._list2D:
                        try:
                            stack = self.__getattribute__(el)
                            # if self.__getattribute__('_ifmask_'+el) and hasattr(self,'_mask_' + el): # sometimes ifmask is true but there is no real mask
                            if hasattr(self, '_mask_' + el):
                                maskname = '_mask_' + el
                                # print(maskname)
                            # if not self.__getattribute__('_ifmask_'+el) or not hasattr(self,'_mask_' + el):
                            if not hasattr(self, '_mask_' + el):  # disregard the _ifmask
                                maskname = 'No_Mask'  # not existent attribute filtered away from the roi_sum function
                                # print(maskname , get_linenumber())
                            # print('_roi_limits_' + el)
                            for pos, roi in enumerate(self.__getattribute__('_roi_limits_' + el), start=0):
                                # sometimes ifmask is true but there is no real mask
                                if maskname != 'No_Mask' and self.__getattribute__('_mask_' + el) != 'No_Mask':
                                    # print(maskname , get_linenumber())
                                    # roiname = self.__getattribute__('_roi_names_' + el)[pos] +'_'+el+'_new'
                                    # Mask Filters Time
                                    roiname1 = 'roi'+str(pos) + '_'+el+'_MFT'
                                    # print(roiname)
                                    self.calcROI(
                                        stack, roi, maskname, self._integration_time, roiname1, coef='default', filt='default')
                                    # Mask only correction
                                    roiname2 = 'roi'+str(pos) + '_'+el+'_M'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)
                                if maskname == 'No_Mask' or self.__getattribute__('_mask_' + el) == 'No_Mask':
                                    print(get_linenumber())
                                    # Mask only correction
                                    roiname2 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)
                                    if roiname2 not in self.attlist:
                                        self.attlist.append(roiname2)

                                    # Mask only correction
                                    roiname3 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly_FT'
                                    self.calcROI(
                                        stack, roi, maskname, roiname3)
                                    if roiname3 not in self.attlist:
                                        self.attlist.append(roiname3)
                        except:
                            if self.verbose != 'NO':
                                print(
                                    'issues with ', el, 'data Matrix/mask/roi_limit', ' Line: ', get_linenumber())
                                # raise
    #                           self.calcROI(self, stack,roiextent, maskname,attcoef, filters, acqTime, ROIname)
        return

    def calcROI_auto2(self):
        '''if exist _coef, _integration_time, _roi_limits, _roi_names it can be applied
        to recalculate the roi on one or more 2D detectors.
        it configure the se of calcROI for multiple detectors/mask 
        filters and motors are shifted of one points for the FLY. corrected in the self.calcROI
        For SBS the data point when the filter is changed is collected with no constant absorber and therefore is rejected.
        if mask exist: correction applied
        if acq time exist correction applied
        if filters exist correction applied '''
        if self.det2d():
            # print('calc 3 ', get_linenumber())
            if self.scantype == 'SBS':  # SBS Correction #######################################
                if hasattr(self, '_npts'):
                    print('Auto ROI already applied')
                # check if the process was alredy runned once on this object
                if not hasattr(self, '_npts'):
                    self._npts = len(self.__getattribute__(self._list2D[0]))

                    for el in self._list2D:
                        try:
                            stack = self.__getattribute__(el)
                            # if self.__getattribute__('_ifmask_'+el) and hasattr(self,'_mask_' + el): # sometimes ifmask is true but there is no real mask
                            if hasattr(self, '_mask_' + el):
                                maskname = '_mask_' + el
                                # print(maskname , get_linenumber())
                            # if not self.__getattribute__('_ifmask_'+el) or not hasattr(self,'_mask_' + el):
                            if not hasattr(self, '_mask_' + el):  # disregard the _ifmask
                                maskname = 'No_Mask'  # not existent attribute filtered away from the roi_sum function
                                # print(maskname , get_linenumber())
                            # print('_roi_limits_' + el)
                            for pos, roi in enumerate(self.__getattribute__('_roi_limits_' + el), start=0):
                                # and self.__getattribute__('_mask_' + el)!='No_Mask': # sometimes ifmask is true but there is no real mask
                                if maskname != 'No_Mask':
                                    # print(maskname , get_linenumber())
                                    # roiname = self.__getattribute__('_roi_names_' + el)[pos] +'_'+el+'_new'
                                    # Mask Filters Time
                                    roiname1 = 'roi'+str(pos) + '_'+el+'_MFT'
                                    # print(roi, '  ', roiname1, '  ', maskname, '  ', get_linenumber())

                                    self.calcROI(
                                        stack, roi, maskname, self._integration_time, roiname1, coef='default', filt='default')
                                    # Mask only correction
                                    roiname2 = 'roi'+str(pos) + '_'+el+'_M'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)

                                # or self.__getattribute__('_mask_' + el)=='No_Mask':
                                if maskname == 'No_Mask':
                                    print(get_linenumber())
                                    # Num only correction
                                    roiname2 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)
                                    # print(get_linenumber())
                                    if roiname2 not in self.attlist:
                                        self.attlist.append(roiname2)
                                        print(get_linenumber())
                                    # Num only correction
                                    roiname3 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly_FT'
                                    # print(roiname3)
                                    # calcROI(self, stack,roiextent, maskname, acqTime, ROIname, coef='default', filt='default')
                                    # if maskname is 'No_Mask' apply only time and Filter
                                    self.calcROI(
                                        stack, roi, maskname, self._integration_time, roiname3)
                                    print(get_linenumber())
                                    if roiname3 not in self.attlist:
                                        self.attlist.append(roiname3)
                        except:
                            if self.verbose != 'NO':
                                print(
                                    'issues with ', el, 'data Matrix/mask/roi_limit', ' Line: ', get_linenumber())

            if self.scantype == 'FLY':  # FLY correction ##########################
                if hasattr(self, '_npts'):
                    print('Auto ROI already applied')
                # check if the process was alredy runned once on this object
                if not hasattr(self, '_npts'):
                    self._npts = len(self.__getattribute__(self._list2D[0]))
                    for el in self._list2D:
                        try:
                            stack = self.__getattribute__(el)
                            # if self.__getattribute__('_ifmask_'+el) and hasattr(self,'_mask_' + el): # sometimes ifmask is true but there is no real mask
                            if hasattr(self, '_mask_' + el):
                                maskname = '_mask_' + el
                                # print(maskname , get_linenumber())
                            # if not self.__getattribute__('_ifmask_'+el) or not hasattr(self,'_mask_' + el):
                            if not hasattr(self, '_mask_' + el):  # disregard the _ifmask
                                maskname = 'No_Mask'  # not existent attribute filtered away from the roi_sum function
                                # print(maskname , get_linenumber())
                            # print('_roi_limits_' + el)
                            for pos, roi in enumerate(self.__getattribute__('_roi_limits_' + el), start=0):
                                # and self.__getattribute__('_mask_' + el)!='No_Mask': # sometimes ifmask is true but there is no real mask
                                if maskname != 'No_Mask':
                                    # print(maskname , get_linenumber())
                                    # roiname = self.__getattribute__('_roi_names_' + el)[pos] +'_'+el+'_new'
                                    # Mask Filters Time
                                    roiname1 = 'roi'+str(pos) + '_'+el+'_MFT'
                                    # print(roiname)
                                    self.calcROI(
                                        stack, roi, maskname, self._integration_time, roiname1, coef='default', filt='default')
                                    # Mask only correction
                                    roiname2 = 'roi'+str(pos) + '_'+el+'_M'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)
                                # or self.__getattribute__('_mask_' + el)=='No_Mask':
                                if maskname == 'No_Mask':

                                    # Num only correction
                                    roiname2 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly'
                                    self.numIntROI(
                                        stack, roi, maskname, roiname2)
                                    # print(get_linenumber())
                                    if roiname2 not in self.attlist:
                                        self.attlist.append(roiname2)

                                    # Num only correction
                                    roiname3 = 'roi' + \
                                        str(pos) + '_'+el+'_NumOnly_FT'
                                    # print(roiname3)
                                    # calcROI(self, stack,roiextent, maskname, acqTime, ROIname, coef='default', filt='default')
                                    # if maskname is 'No_Mask' apply only time and Filter
                                    self.calcROI(
                                        stack, roi, maskname, self._integration_time, roiname3)
                                    if roiname3 not in self.attlist:
                                        self.attlist.append(roiname3)
                        except:
                            if self.verbose != 'NO':
                                print(
                                    'issues with ', el, 'data Matrix/mask/roi_limit', ' Line: ', get_linenumber())
                                # raise
#                           self.calcROI(self, stack,roiextent, maskname,attcoef, filters, acqTime, ROIname)

    def roishow(self, roinumber, detectorname, imageN=1):
        '''Image number is the image position in the stack series of detector name and roinumber is ROI line in the 
        _roi_name variable
        roinumber can also be desired region measured or in the form :[xpos, ypos,  dx,  dy]
        '''

        if isinstance(roinumber, int):
            for el in self.attlist:
                if '_roi_limits_'+detectorname in el:
                    roi_limits = self.__getattribute__(el)
                    roi = roi_limits[roinumber]
                    stack = self.getStack(detectorname)
        if isinstance(roinumber, list):
            roi = roinumber
            stack = self.getStack(detectorname)

        plt.imshow(stack[imageN][roi[1]:roi[1]+roi[3], roi[0]:roi[0] +
                   roi[2]], norm=LogNorm(vmin=0.01, vmax=1e5), cmap='jet')
        return

    def prj(self, axe=0, mask_extra=None):
        '''Project the 2D detector on the coosen axe of the detector and return a matrix 
        of size:'side detector pixels' x 'number of images' 
        axe = 0 ==> x axe detector image
        axe = 1 ==> y axe detector image
        specify a mask_extra variable if you like. 
        Mask extra must be a the result of np.load(YourMask.npy)'''
        if hasattr(self, 'mask'):
            mask = self.__getattribute__('mask')
        if not hasattr(self, 'mask'):
            mask = 1
        if np.shape(mask_extra):
            mask = mask_extra
            if np.shape(mask) == (240, 560):
                self.make_maskFrame_xpad()
                mask = mask  # & self.mask0_xpad
        for el in self.attlist:
            bla = self.__getattribute__(el)
            # get the attributes from list one by one
            # check for image stacks Does Not work if you have more than one 2D detectors
            if len(bla.shape) == 3:
                mat = []
                if np.shape(mask) != np.shape(bla[0]):  # verify mask size
                    print(np.shape(mask), 'different from ',
                          np.shape(bla[0]), ' verify mask size')
                    mask = 1
                for img in bla:
                    if np.shape(mat)[0] == 0:  # fill the first line element
                        mat = np.sum(img ^ mask, axis=axe)
                    if np.shape(mat)[0] > 0:
                        mat = np.c_[mat, np.sum(img ^ mask, axis=axe)]
                # generate the new attribute
                setattr(self, str(el+'_prjX'), mat)

    def det2d(self):
        '''it retunrs the name/s of the 2D detector'''
        list2D = []
        for el in self.attlist:
            bla = self.__getattribute__(el)
            # print(el, bla.shape)
            # get the attributes from list one by one
            if isinstance(bla, (np.ndarray, np.generic)):
                if len(bla.shape) == 3:  # check for image stacks
                    list2D.append(el)
        if len(list2D) > 0:
            # self._list2D = list2D
            return list2D
        else:
            return False

    def addmetadata(self, hdf5address, attrname):
        '''It goes back to the nxsfile looking for the metadata stored in the \SIXS part of the file.
        USE:
        toto.addmetadata('i14-c-cx1-ex-fent_v.7-mt_g','ssl3vg') 
        it will add new attributes with the rootname 'attrname'  '''

        ff = tables.open_file(self.directory + self.filename, 'r')
        f = ff.list_nodes('/')[0]
        toto = f.SIXS.__getattr__(hdf5address)
        for leaf in toto:
            try:
                list.append(self.attlist,  (attrname + '_' + leaf.name))
                self.__dict__[(attrname + '_' + leaf.name)] = leaf[:]
                print(leaf.name)
            except:
                print(
                    'Issue with the hdf5 address, there might be a too complex structure')
            # print(leaf[:])
        ff.close()

    def save2txt(self, list2exp):
        '''It saves a txt file with the variable listed.
        the variable must be present in the "attlist" attribute
        it expect a list of strings. 
        It dos not export images
        '''
        filesave = self.filename[:-4] + '_extr.txt'
        longname = os.path.join(self.directory, filesave)
        self.calcROI_auto2()
        outdata = np.empty(self._npts)
        exported = []

        for el in list2exp:
            if el in self.attlist:
                if len(np.shape(self.__getattribute__(el))) > 1 and (self.verbose == 'YES'):
                    print('attribute ', el, 'has more that one dimension')
                if len(np.shape(self.__getattribute__(el))) == 1:
                    try:
                        # print('adding ', el)
                        loc = self.__getattribute__(el)
                        # print(loc)
                        outdata = np.c_[outdata, loc]
                        exported.append(el)
                    except:
                        raise

        headerline = '   '.join(exported)
        np.savetxt(longname, outdata[:, 1:], delimiter='\t', header=headerline)

    def saveXY(self, list2exp):
        '''It saves a txt file with the X Y variables listed ['x','y'].
        it removes the "holes" of the unknown filters positions
        the variable must be present in the "attlist" attribute
        it expect a list of strings. 
        It dos not export images
        '''
        filesave = self.filename[:-4] + '_extr.txt'
        longname = os.path.join(self.directory, filesave)
        self.calcROI_auto2()
        # x = np.empty(self._npts)
        # y = np.empty(self._npts)
        data = np.empty(self._npts)
        exported = []
        x = self.__getattribute__(list2exp[0])
        y = self.__getattribute__(list2exp[1])
        # for el in list2exp:
        #     if el in self.attlist:
        #         if len(np.shape(self.__getattribute__(el))) > 1 and (self.verbose == 'YES'):
        #                print('attribute ',el, 'has more that one dimension' )
        #         if len(np.shape(self.__getattribute__(el))) == 1:
        #             try:
        #                 #print('adding ', el)
        #                 loc = self.__getattribute__(el)
        #                 #print(loc)
        #                 data = np.c_[data, loc]
        #                 exported.append(el)
        #             except:
        #                 raise

        # print (y.shape)
        yout = y[y > 0.5]
        xout = x[y > 0.5]
        outdata = np.c_[xout, yout]
        # print(outdata[:,1:])
        headerline = '   '.join(list2exp)
        np.savetxt(longname, outdata, delimiter='\t', header=headerline)

    def det2jpg(self, det, st, nd):
        '''It export  the images fo the "detector as jpg. The numerical range is 
        from st to nd'''
        imagesN = np.arange(st, nd)
        if hasattr(self, det):

            for el in imagesN:
                filesave = self.filename[:-4] + '_imgn_'+str(el)+'.jpg'
                longname = os.path.join(self.directory, filesave)
                fig2save = plt.imshow(getattr(self, det)[el])
                fig2save.figure.savefig(longname)
