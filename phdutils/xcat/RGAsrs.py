# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 13:07:25 2014

@author: andrea
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pylab as pl
import time
import datetime
from phdutils.sixs import utilities3 as ut3

from collections import namedtuple
'''It is meant to read/work on files from the MKS mass spectrometer and NOT from the SRS
it returns: the data matrix, the unix time at the start of the scan, the dictionary linking column and mass 
 
'''

#Range = namedtuple("Range", ["start", "duration"])


def RgaLoad(filename):
    '''Loads file from the the SRS mass spec 
    it returns:
    - the data matrix 
    - the unix time at the start of the scan
    - the dictionary linking column and mass '''

    # here the strings to search in the header of the file as reference points
    HeaderST = 'Channel,  Mass(amu),'
    HeaderEND = 'Time(s) '
    sttime = 'Start time, '
    channel = []
    masses = []
    with open(filename) as myFile:
        hst = 1000
        # hend=1000
        for num, line in enumerate(myFile, 1):
            if sttime in line:
                timeline = line.rstrip()  # rstrip remove the spaces at the end of the line
                #print( timeline[12:])
                starttime = time.mktime(datetime.datetime.strptime(
                    timeline[12:], "%b %d, %Y  %I:%M:%S  %p").timetuple())
                # print type(starttime)
            if HeaderST in line:
                hst = num
                # print hst
            if num > hst:
                # read the line and test for the Hend
                LineFields = line.split()
                # if first field of "line shape" > 2 AKA exclude empty lines
                if np.shape(LineFields)[0] > 2:
                    if HeaderEND not in line:  # exclude from the dictionary the data column header used as reference point while searching through the file
                        channel.append(int(float(LineFields[0])))
                        masses.append(int(float(LineFields[1])))
                if HeaderEND in line:
                    hend = num
                    # print hend
                    dictionary = dict(list(zip(channel, masses)))
                    # print dictionary
                    break

            # print line

        # print 'found at line:', num2
        # for nu, line in enumerate(myFile, 1):
        #    if nu > hst & nu < hend:     # analyse the header
        #        print line

    # data=pd.read_csv(filename,delimiter=',',skiprows=hend)
    data = np.genfromtxt(filename, dtype='float',
                         comments='#', delimiter=',', skip_header=hend+1)
    return data, starttime, dictionary


def RgaPlot(filename, masses, TimeRange=None, style='none'):
    '''it plots the raw data as function of the time in seconds since the start of the experiment
    using the option time=[start, deltaTime] permet to select only the dat within that range
    masses is a vector of integer numbers e.g. [28,32,40]
    aaaand it had the XKCD style option... ;-)
    Time range is defined as start time + interval eg: [1434112800,11000]    
    '''

    plt.figure(figsize=(18, 15), dpi=80, facecolor='w', edgecolor='k')
    data, starttime, massDict = RgaLoad(filename)
    #import re
    # data,sttime,massDict=RgaLoad(filename)
    tt = data[:, 0]+starttime  # time in  epoch of the RGA data
    # timeEXP=tt-TimeRange[0] # experiment time used in the plotting; giving an x axe starting at 0
    timeEXP = tt
    mingra = 1
    maxgra = 0
    if TimeRange is None:
        print('No Time window')
        timeEXP = tt-min(tt)
        Dstart = 0
        Dend = -1
    else:
        print('Time Window')
        Dstart = ut3.find_nearest(tt, TimeRange[0])[0]
        #Dstart = min(matplotlib.mlab.find(tt > TimeRange[0]))
        timeEXP = tt-TimeRange[0]
        #print (Dstart, time[0])
        # print 'here'
        timeEND = TimeRange[0] + TimeRange[1]
        #Dend = max(matplotlib.mlab.find(tt<timeEND))
        Dend = ut3.find_nearest(tt, timeEND)[0]
    for el in masses:
        #name='Mass '+str(el)
        col = list(massDict.keys())[list(massDict.values()).index(el)]
        vect = np.abs(data[:, col])
        miny = min(vect)
        maxy = max(vect)
        if mingra > miny:
            mingra = miny

        if maxgra < maxy:
            maxgra = maxy

        if style != 'XKCD':
            # fig1=plt.figure()
            #print (timeEXP[Dstart])
            plt.plot(timeEXP[Dstart:Dend], vect[Dstart:Dend],
                     label='Mass '+str(el))

        elif style == 'XKCD':
            with plt.xkcd():
                figXKCD = plt.figure()
                figXKCD.plot(timeEXP[Dstart:Dend],
                             vect[Dstart:Dend], label='Mass '+str(el))

    # plt.gcf().autofmt_xdate()
    # plt.show()
    # plt.tight_layout()

    # pl.xlim(0,TimeRange[1])
    pl.xlim(0, max(timeEXP))
    #print (mingra, maxgra)
    pl.ylim(mingra-mingra*0.05, maxgra+maxgra*0.05)
    pl.semilogy()
    pl.xlabel('Time (s)', fontsize=16)
    pl.ylabel('Partial Pressure (mBar)', fontsize=16)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=5, mode="expand", borderaxespad=0.)
    # plt.legend(loc=2)
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.grid(b=True, which='minor', color='r', linestyle='--')
    plt.show()


def RgaSelect(filename, masses, TimeRange=None):
    '''it returns the raw data as function of the time in seconds since the start of the experiment
    using the option time=[start, deltaTime]. It permits to select only the data within that range
    masses is a vector of integer numbers e.g. [28,32,40]'''
    data, starttime, massDict = RgaLoad(filename)
    if TimeRange is None:
        #print ('No Time window')
        TimeRange = [starttime, (data[-1, 0]-data[0, 0])]
        #Dstart = 0
        #Dend = -1
    #import re
    # data,sttime,massDict=RgaLoad(filename)
    tt = data[:, 0]+starttime  # time in  epoch of the RGA data
    vectAll = tt-TimeRange[0]
    # mingra=1
    # maxgra=0
    if TimeRange is None:
        #print ('No Time window')
        Dstart = 0
        Dend = -1
    else:
        #print ('Time Window')
        Dstart = ut3.find_nearest(tt, TimeRange[0])[0]
        #print (Dstart, time[0])
        # print 'here'
        timeEND = TimeRange[0] + TimeRange[1]
        Dend = ut3.find_nearest(tt, timeEND)[0]

    for el in masses:
        #name='Mass '+str(el)
        col = list(massDict.keys())[list(massDict.values()).index(el)]
        vect = data[:, col]
        vectAll = np.c_[vectAll, vect]

    return vectAll[Dstart:Dend, :]


def RgaNormPlot(filename, masses, TimeRange=None, plot=None):
    '''It plots the RGA signals normalised to the first mass of the list, often the carrier Ar. '''

    vectAll = RgaSelect(filename, masses, TimeRange=TimeRange)
    vectAllnorm = vectAll[:, 0]
    #    print type(vectAllnorm)
    for index, el in enumerate(masses[1:], start=2):
        # start = 2 because on the batches matrix the column 0 is the time col 1 is the carrier gas
        y = vectAll[:, index]/vectAll[:, 1]
        vectAllnorm = np.c_[vectAllnorm, y]
        # print np.shape(y)

        if plot != None:
            plt.plot(vectAllnorm[:, 0], y, label='Mass '+str(el))

        pl.semilogy()
        pl.xlabel('Time (s)', fontsize=16)
        pl.ylabel('Partial Pressure A.U', fontsize=16)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=5, mode="expand", borderaxespad=0.)
        # plt.legend(loc=2)
        plt.grid(b=True, which='major', color='b', linestyle='-')
        plt.grid(b=True, which='minor', color='r', linestyle='--')
        plt.show()

    return vectAllnorm


def RgaNormPtotPlot(filename, masses, Pres, TimeRange=None, plot=None):
    '''It plots the RGA signals normalised to the sum of the masses list. 
    and rescaled to the total imposed pressure of the masses sum
    It can also be used just to select and normalise to the total pressure'''
    vectAll = RgaSelect(filename, masses, TimeRange=TimeRange)
    gasses = vectAll[:, 1:]
    totP = gasses.sum(axis=1)/Pres
    vectAllnorm = vectAll[:, 0]
    for index, el in enumerate(masses):
        vectAllnorm = np.c_[vectAllnorm, gasses[:, index]/totP]

    plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

    if plot != 'NO':
        for index, el in enumerate(masses):
            plt.plot(vectAllnorm[:, 0], vectAllnorm[:,
                     index+1], linewidth=2, label='Mass '+str(el))
        pl.semilogy()
        plt.legend(loc=2, fontsize=15)
        pl.xlabel('Time (s)', fontsize=16)
        pl.ylabel('Partial Pressure (Bar)', fontsize=16)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=5, mode="expand", borderaxespad=0.)
        # plt.legend(loc=2)
        plt.grid(b=True, which='major', color='b', linestyle='-')
        plt.grid(b=True, which='minor', color='r', linestyle='--')
        plt.show()

    return vectAllnorm
