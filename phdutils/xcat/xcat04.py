# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:46:30 2015

@author: Andrea & Jean-Baptiste
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab as pl
import time
import datetime
import os
import RGAsrs
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from phdutils.sixs import utilities3 as ut3


class emptyO(object):
    '''Empty class used as container in the nxs2spec case. '''
    pass


class DataSet(object):
    '''Dataset read the file and store it in an object, from this object we can 
    retrive the data to use it: 
    Use as:
    dataObject = nxsRead3.DataSet( path/filename, path )
    filename can be /path/filename or just filename
    directory is optional and it can contain /path/path
    both are meant to be string
    Meant to be used on the data produced after the 11/03/2019 data of the 
    upgrade of the datarecorder'''

    def __init__(self, directory, filename):
        self.longname = os.path.join(directory, filename)
        self.filename = filename
        self.data = np.genfromtxt(
            self.longname, dtype=float, skip_header=1, delimiter='')

    def _XCATtime(self, TimeVector, TimeRange=None):
        '''Tranform the xcat time into unix epoch, and eventually select a desired 
        range using the option TimeRange=[start, deltaTime]. deltaTima aka duration
        ### Mind that there is a different time vector for every valiable stored ####
        '''
        fullEpoch = TimeVector - \
            2082844800  # transform the xcat epoch (epoch labview time) in unix epoch
        if not TimeRange:
            # print 'No Time window'
            Dstart = 0  # vectorTimeXCAT[0]
            Dend = -1  # vectorTimeXCAT[-1]
            return Dstart, Dend, fullEpoch[0:-1]
        else:
            # print 'Time Window'
            # min(matplotlib.mlab.find(tt > TimeRange[0])) #TimeRange[0]=start
            Dstart, Dstart_value = ut3.find_nearest(fullEpoch, TimeRange[0])
            timeEND = TimeRange[0] + TimeRange[1]  # TimeRange[1]=duration
            # max(matplotlib.mlab.find(tt<timeEND))
            Dend, Dend_value = ut3.find_nearest(fullEpoch, timeEND)
            if len(fullEpoch[Dstart:Dend]) < 1:
                print('######   No Epoch/Timerange Overlap     #######')
            # position within the vector
            return Dstart, Dend, fullEpoch[Dstart:Dend]
        pass

    def selectGas(self, TimeRange=None):
        ''' For a given file build 9 vectors/attributes (NO, H2, O2, CO, Ar, Valves, Shunt, 
        Reactor, Drain) containings the logged data from the xcat software. 
        If desired selects the data in a range TimeRange=[start, deltaTime]
        it also sets the beginning of the time window to 0
        '''

        # might wanna check nametuple to spit out an object with the output as attributes
        NO = np.array([])
        H2 = np.array([])
        O2 = np.array([])
        CO = np.array([])
        Ar = np.array([])
        Shunt = np.array([])
        Reactor = np.array([])
        Drain = np.array([])
        Valves = np.array([])

        # define the position of Dstart and Dend
        NOst, NOnd, NO = self._XCATtime(self.data[:, 0], TimeRange=TimeRange)
        # NO=self.data[NOint[0]:NOint[1], 0] - 2082844800 -TimeRange[0]   #select the time range between Dstart et Dend
        NO = np.c_[NO, self.data[NOst:NOnd, 1]]  # flow Range
        NO = np.c_[NO, self.data[NOst:NOnd, 2]]  # setpoint  Range
        NO = np.c_[NO, self.data[NOst:NOnd, 3]]  # valve Range
        self.NO = NO

        # select the time range between Dstart et Dend
        H2st, H2nd, H2 = self._XCATtime(self.data[:, 4], TimeRange=TimeRange)
        H2 = np.c_[H2, self.data[H2st:H2nd, 5]]  # flow Range
        H2 = np.c_[H2, self.data[H2st:H2nd, 6]]  # setpoint  Range
        H2 = np.c_[H2, self.data[H2st:H2nd, 7]]  # valve Range
        self.H2 = H2

        # select the time range between Dstart et Dend
        O2st, O2nd, O2 = self._XCATtime(self.data[:, 8], TimeRange=TimeRange)
        O2 = np.c_[O2, self.data[O2st:O2nd, 9]]  # flow Range
        O2 = np.c_[O2, self.data[O2st:O2nd, 10]]  # setpoint  Range
        O2 = np.c_[O2, self.data[O2st:O2nd, 11]]  # valve Range
        self.O2 = O2

        # select the time range between Dstart et Dend
        COst, COnd, CO = self._XCATtime(self.data[:, 12], TimeRange=TimeRange)
        CO = np.c_[CO, self.data[COst:COnd, 13]]  # flow Range
        CO = np.c_[CO, self.data[COst:COnd, 14]]  # setpoint  Range
        CO = np.c_[CO, self.data[COst:COnd, 15]]  # valve Range
        self.CO = CO

        # select the time range between Dstart et Dend
        Arst, Arnd, Ar = self._XCATtime(self.data[:, 16], TimeRange=TimeRange)
        Ar = np.c_[Ar, self.data[Arst:Arnd, 17]]  # flow Range
        Ar = np.c_[Ar, self.data[Arst:Arnd, 18]]  # setpoint  Range
        Ar = np.c_[Ar, self.data[Arst:Arnd, 19]]  # valve Range
        self.Ar = Ar

        # select the time range between Dstart et Dend
        Shuntst, Shuntnd, Shunt = self._XCATtime(
            self.data[:, 20], TimeRange=TimeRange)
        Shunt = np.c_[Shunt, self.data[Shuntst:Shuntnd, 21]]  # flow Range
        Shunt = np.c_[Shunt, self.data[Shuntst:Shuntnd, 22]]  # setpoint  Range
        Shunt = np.c_[Shunt, self.data[Shuntst:Shuntnd, 23]]  # valve Range
        self.Shunt = Shunt

        # select the time range between Dstart et Dend
        Reactorst, Reactornd, Reactor = self._XCATtime(
            self.data[:, 24], TimeRange=TimeRange)
        Reactor = np.c_[
            Reactor, self.data[Reactorst:Reactornd, 25]]  # flow Range
        # setpoint  Range
        Reactor = np.c_[Reactor, self.data[Reactorst:Reactornd, 26]]
        # valve Range
        Reactor = np.c_[Reactor, self.data[Reactorst:Reactornd, 27]]
        self.Reactor = Reactor

        # select the time range between Dstart et Dend
        Drainst, Drainnd, Drain = self._XCATtime(
            self.data[:, 28], TimeRange=TimeRange)
        Drain = np.c_[Drain, self.data[Drainst:Drainnd, 29]]  # flow Range
        Drain = np.c_[Drain, self.data[Drainst:Drainnd, 30]]  # setpoint  Range
        Drain = np.c_[Drain, self.data[Drainst:Drainnd, 31]]  # valve Range
        self.Drain = Drain

        # select the time range between Dstart et Dend
        Valvesst, Valvesnd, Valves = self._XCATtime(
            self.data[:, 32], TimeRange=TimeRange)
        Valves = np.c_[Valves, self.data[Valvesst:Valvesnd, 33]]  # MIX
        Valves = np.c_[Valves, self.data[Valvesst:Valvesnd, 34]
                       ]                     # MRS
        # Valve=np.c_[Ar,self.data[Valvest:Valvend],35]                     # MRX
        self.Valves = Valves

    def getSection(self, TimeRange, mixPos=3, mrsPos=10):
        # def getSection(GasLog, mixPos=3, mrsPos=10, TimeRange=None):
        '''It looks for the batch or flow phases by searching for the desired couple of values for mrs and mix...
        It returns the point in time epoch when the mixPos, mrsPos are verified.
        TimeRange = [Epoch start, duration]'''
        self.selectGas(TimeRange)
        mrs = self.Valves[:, 2]
        mix = self.Valves[:, 1]
        timePhase = self.Valves[:, 0]

        # here we search only the points where there is a variation in the mrs state
        mrsd = matplotlib.mlab.find(np.diff(mrs) != 0)

        SectionStarts = []
        #print('inter is: ',inter)
        for el in mrsd:
            #print('element is: ',el)
            # take few data points after the valve movements since the valve movement is not immediate nor constant in shift
            if (np.average(mrs[el+3:el+6]) == mrsPos) & (np.average(mix[el+3:el+6]) == mixPos):
                # This if do not exclude to have two values one after the others registered while the valve is moving
                SectionStarts.append(timePhase[el])
        # rejecting the false positives induced by the valve movements on the base of the
        # temporal shifts between selected points

        # print(SectionStarts)
        SectionStartShort = []
        for index, el in enumerate(np.diff(SectionStarts), start=0):
            if el < 20:  # 3 was the walue used from 2016 to  2018
                SectionStartShort.append(SectionStarts[index+1])
            elif el > 20:
                SectionStartShort.append(SectionStarts[index])

        SectionStartShort = np.asanyarray(SectionStartShort)
        SectionStartShort = SectionStartShort + TimeRange[0]
        # reject the double elemnts
        SectionStartShort = np.unique(SectionStartShort)
        # print(SectionStartShort)
        #self.Section = SectionStartShort
        return SectionStartShort

    ################## Re-implemented until here 22 10 2020   ##########################
    def Gas7Plot(self, filenameXCAT, filenameRGA, masslist, channelList, TimeRange=None):
        '''It generate a plot with 8 subplots containing:
        1) Flow measures contained in channelList e.g. [CO,O2,Ar]
        2) Flow set points
        3) Valve position
        4) MRS/MIX position 
        5) Reactor measured pressure
        6) Reactor Valve position
        7) RGA gasses contained in mass list e.g. [40,28,32]
        8) RGA gasses normalised on the First signal of MassList
        here an example on how to use it:
        xcat02.Gas8Plot('20150612-110620.txt','FlowVsLeakTest07.txt',[40,28,32,44],['O2','CO','Ar','shu'],TimeRange=[1434112800,11000])
        '''

        plt.figure(num=None, figsize=(11, 15), dpi=80,
                   facecolor='w', edgecolor='k')
        # GasLog=self.selectGas(TimeRange=TimeRange)

        channelpos = list()
        channels = ('NO', 'H2', 'O2', 'CO', 'Ar',
                    'Shunt', 'Reactor', 'Drain', 'Valves')
        colors = ('-c', '-r', '-g', '-k', 'm', '-b', '-k', '-b')

        # find wich channels we use
        for el in channelList:
            if np.isreal(channels.index(el)):
                channelpos.append(channels.index(el))
            else:
                print('At least one channel is not contained in the data names')

        # Plot 1
        '''1) Flow measures
        '''
        plt.subplot(711)
        yy = np.array([])
        for el, index_el in zip(channelList, channelpos):
            flow_data = getattr(self, el)
            plt.plot(flow_data[:, 0], flow_data[:, 1],
                     colors[index_el], label=channels[index_el])
            yy = np.concatenate([yy, flow_data[:, 1]])

        # pl.xlim(0,TimeRange[1])
        # pl.ylim(-5, yy.max()+5)
        pl.ylabel('$Flow$', fontsize=16)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=5, mode="expand", borderaxespad=0.)

        # Plot 2
        '''2) Flow setpoints
        '''
        plt.subplot(712)
        yy = np.array([])
        for el, index_el in zip(channelList, channelpos):
            flow_data = getattr(self, el)
            plt.plot(flow_data[:, 0], flow_data[:, 2],
                     colors[index_el], label=channels[index_el])
            yy = np.concatenate([yy, flow_data[:, 2]])
        # pl.xlim(0,TimeRange[1])
        # pl.ylim(-5, yy.max()+5)
        pl.ylabel('$Setpoint$', fontsize=16)

        # Plot 3
        '''3) Valves Positions
        '''
        plt.subplot(713)
        yy = np.array([])

        for el, index_el in zip(channelList, channelpos):
            flow_data = getattr(self, el)
            plt.plot(flow_data[:, 0], flow_data[:, 3],
                     colors[index_el], label=channels[index_el])
            yy = np.concatenate([yy, flow_data[:, 3]])

        pl.xlim(0, TimeRange[1])
        pl.ylim(yy.min()-1, yy.max()+1)
        pl.ylabel('$Valve$', fontsize=16)

        # Plot 4
        '''4) MRS/MIX Position
        '''
        plt.subplot(714)

        yy = np.concatenate([GasLog[5][:, 2], GasLog[5][:, 1]])

        plt.plot(GasLog[5][:, 0], GasLog[5][:, 2], '-b', label='MRS')
        plt.plot(GasLog[5][:, 0], GasLog[5][:, 1], '-g', label='MIX')
        pl.xlim(0, TimeRange[1])

        pl.ylim(yy.min()-1, yy.max()+1)
        pl.ylabel('$MRS/MIX$', fontsize=16)

    #     plt.subplot(715)  #### Plot 5
    #     '''5) Reactor pressure
    #     '''

    #     plt.plot(GasLog[7][:,0], GasLog[7][:,1], '-b')
    #     pl.xlim(0,TimeRange[1])
    #     pl.ylim(0, 1.2)
    #     pl.ylabel('$Reactor Pressure$',fontsize=16)

    #     plt.subplot(716)  #### Plot 6
    #     '''5) Reactor valve
    #     '''
    #     plt.plot(GasLog[7][:,0], GasLog[7][:,3], '-b' )
    #     pl.xlim(0,TimeRange[1])
    #     pl.ylim(GasLog[7][:,3].min()-.5,GasLog[7][:,3].max()+0.5)
    #     pl.ylabel('$ReactorValve$',fontsize=16)

    # #    plt.subplot(817) #### Plot 7
    # #    '''6) RGA gas composition
    # #    '''
    # #    RGAsrs.RgaPlot(filenameRGA, masslist,TimeRange=TimeRange)

    #     plt.subplot(717) #### Plot 8
    #     '''RGA gas signals normalised on the first gas of masslist
    #     '''
    #     RgaGas=RGAsrs.RgaSelect(filenameRGA, masslist, TimeRange=TimeRange)

    #     yy=np.array([])

    #     for ind, el in enumerate(masslist[1:], start=2):

    #         plt.plot(RgaGas[:,0], RgaGas[:,ind]/RgaGas[:,1], colors[ind-1])
    #         yy=np.concatenate([yy,RgaGas[:,ind]/RgaGas[:,1]])

    #     pl.xlim(0,TimeRange[1])
    #     pl.ylim(yy.min(), yy.max())
    #     pl.ylabel('Normalised Int (A.U.)',fontsize=16)
    #     pl.xlabel('Time (s)', fontsize=16)
    #     pl.legend(loc=1)
    plt.show()


def getChange(timeV, vect):
    '''Return a vector containing the timeV and position when the variation occur of the in input vector
    meant to search on the setpoint vectors'''
    positions = np.where(vect[:-1] != vect[1:])[0]
    times = timeV[positions]
    return positions, times


def gasScalingFactor(self, filenameXCAT, timePoints, TimeRange=None, corrF_Ar=None, corrF_Flux=None):
    '''return the scaling factor for RGA masses normalised to the Ar signal.
    In other words the imposed partial pressure of the carrier gas.
    It assumes that the gases are flown as pure through the mass flow controller, 
    If using mixed gas the easiest way is to insert a manual corrF_Ar, corrF_Flux factors, .
    It returns a scaling factor for each time point 
    (timepoints is a list so must be inserted as [10,20,30, ..., ... ])
    if TimeRange is shified the timepoints are expressed in time elapsed from the
    start of the selected timewindow'''

    GasLog = self.selectGas(filenameXCAT, TimeRange=TimeRange)
# select the gasses and reactor from the selectgas output:
# NO, H2, O2, CO, Ar, Valves, Shunt, Reactor, Drain
    GasLogPos = [0, 1, 2, 3, 4, 7]
    shift = 15  # delay time between batch start and flow stop

    ScalFact = []
    for el1 in timePoints:
        # search eact time point for each gas and the reactor pressure
        TotFlux = 0
        for el2 in GasLogPos[0:5]:
            tt = GasLog[el2][:, 0]
            Dstart = min(matplotlib.mlab.find(tt > el1+shift))
            #print (GasLog[el2][Dstart,2])

            # 1) calculate the total flux and the Ar flux
            # I read the set value!!!!
            TotFlux = TotFlux + GasLog[el2][Dstart, 2]
        #print (TotFlux)
        # retrive only the Ar flux
        Dar = min(matplotlib.mlab.find(GasLog[4][:, 0] > el1+shift))
        ArFlux = GasLog[4][Dar, 2]
#        ArPartPress =
#        print (ArPartPress)

        # 2) retrive the reactor pressure
        DRea = min(matplotlib.mlab.find(GasLog[7][:, 0] > el1+shift))
        ReaPre = GasLog[7][DRea, 2]

        # 3) calculate the factor
        if corrF_Ar == None and corrF_Flux == None:
            fact = ReaPre*ArFlux/TotFlux
        elif corrF_Ar != None or corrF_Flux != None:
            TotFlux = TotFlux + corrF_Flux
            ArFlux = ArFlux + corrF_Ar
            fact = ReaPre*ArFlux/TotFlux

        #print (fact)
        ScalFact.append(fact)
    return ScalFact


def gasSlopes(Xcatfile, RgaFile, masses, shift=15, inter=30, TimeRange=None, savefile='NO', corrF_Ar=None, corrF_Flux=None):
    '''Load the masses e.g. [40,28,32,44] from RGA/XCAt files for the desired experiment and in the 
    selected time range it looks for the batch phases and fits the first 20 seconds of each batch phase 
    with a linear interpolation returning the slope for each gas-batch session.

    It also normalises the gas carrier that MUST be the FIRST MASS OF THE LIST (gas Ar often) 
    and report the partial pressures to the total pressure in the reactor. If it is not Ar check the 
    "gasScalingFactor" function.
    It calls indeed RGAsrs.RgaNormPlot

    it assumes that during the batch phase the flux is not changed but simply goes to the drain.

    The resulta are: 
    Column 1 ==> mass 1 slopes
    Column 2 ==> mass 2 slopes

    slopes are in bar/sec'''

    BatchStarts = getSection(
        Xcatfile, mixPos=3, mrsPos=10, TimeRange=TimeRange)
    BatchStarts = BatchStarts - TimeRange[0]
    print(BatchStarts)
    ScaleFactors = gasScalingFactor(
        Xcatfile, BatchStarts, TimeRange=TimeRange, corrF_Ar=corrF_Ar, corrF_Flux=corrF_Flux)
    #print (ScaleFactors)
    batches = []
    # shift = 15 # delay time between batch start and flow stop
    for el in BatchStarts:
        el = el + TimeRange[0]
        # define the interval to fit: [starting point, delta]
        Interval = [el+shift, inter]
#        print(Interval)
        # ==> computationally poor step, I have to read the file n times
        batch = RGAsrs.RgaSelect(RgaFile, masses, TimeRange=Interval)
#        print (np.shape(batch))
        # batches is a list of matrices, each matrix is a time
        batches.append(batch)
        # column and RGA data referring to a desired mass at the beginning of the (batch) phase
# here we go in for the slope for each gas.
#    return batches

    slopebatches = []
    # 1st loop goes over each batch phase (using the lenght of batchstarts)
    for index1, el in enumerate(BatchStarts):
        slopeMasses = []
        slopeError = []
# The carrier gas should be the 1st listed in the mass list
        ycarrier = batches[index1][:, 1]
        #print (ycarrier.mean())
        for index2, el in enumerate(masses[1:], start=2):
            # start = 2 because on the batches matrix the column 0 is the time col 1 is the carrier gas
            # start from mass 1 because because mass 0 is the carrier gas

            x = batches[index1][:, 0]
            y = batches[index1][:, index2]
#            print (index2, 'index2')
#            print (batches[index1][1,index2])
            ynorm = (y/ycarrier)*ScaleFactors[index1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x, ynorm)
#            plt.plot(x,ynorm,'-p')    # uncomment this line to see the fit
#            yfit=x*slope+intercept    # uncomment this line to see the fit
#            plt.plot(x,yfit)          # uncomment this line to see the fit
            slopeMasses.append(slope)
            slopeError.append(std_err)
        slopeMasses.append(BatchStarts[index1])
        slopeError.append(BatchStarts[index1])
        if index1 == 0:
            #
            #print (np.shape(slopeMasses))
            #print (slopeError.shape)
            #print (np.shape(BatchStarts))
            #slopeMasses = np.append(slopeMasses,BatchStarts,1)
            slopebatches = slopeMasses
            if savefile != 'NO':
                line1 = '#   '
                FileSaveName = RgaFile[:-4]+'_S'+str(shift)+'_slopes.txt'
                FileSaveNameErr = RgaFile[:-4]+'_S'+str(shift)+'_slopesErr.txt'
                f = open(FileSaveName, 'w+')
                g = open(FileSaveNameErr, 'w+')
                for el in masses[1:]:
                    name = 'Mass '+str(el)
                    line1 = str(line1) + name + '   '
                line1 = line1 + 'time' + '   '
                f.writelines(line1+'\n')
                g.writelines(line1+'\n')
                f.writelines('   '.join('%5g' % e for e in slopeMasses) + '\n')
                g.writelines('   '.join('%5g' % e for e in slopeError) + '\n')
        else:
            #        A = numpy.vstack([A, newrow])
            slopebatches = np.vstack([slopebatches, slopeMasses])
            # print(np.shape(slopebatches))
            if savefile != 'NO':
                f.writelines('   '.join('%5g' % e for e in slopeMasses) + '\n')
                g.writelines('   '.join('%5g' % e for e in slopeError) + '\n')
    f.close()
    g.close()

    return BatchStarts, slopebatches


# , corrF_Ar = None, corrF_Flux = None):
def gasSlopes3(Xcatfile, RgaFile, masses, Pres, shift=15, inter=30, TimeRange=None, savefile='NO'):
    '''Load the masses e.g. [40,28,32,44] from RGA/XCAt files for the desired experiment and in the 
    selected time range it looks for the batch phases and fits the first 20 seconds of each batch phase 
    with a linear interpolation returning the slope for each gas-batch session.

    It also normalises all to the gas sum, imposed to be the pressure in the reactor.
    The Mass List Must contains All the gasses injected/produced in the reactor!!!
    The first mass must be nevertheless the carrier Gas (aka Argon)
    It calls indeed RGAsrs.RgaNormPtotPlot instead of RgaNormPlot

    it assumes that during the batch phase the flux is not changed but simply goes to the drain.

    The resulta are: 
    Column 1 ==> mass 1 slopes
    Column 2 ==> mass 2 slopes

    slopes are in bar/sec'''
    BatchStarts = getSection(
        Xcatfile, mixPos=3, mrsPos=10, TimeRange=TimeRange)
    BatchStarts = BatchStarts - TimeRange[0]
    batches = []
    for el in BatchStarts:
        el = el + TimeRange[0]
        # define the interval to fit: [starting point, delta]
        Interval = [el+shift, inter]
        # RGAsrs.RgaSelect(RgaFile, masses,TimeRange=Interval) ### ==> computationally poor step, I have to read the file n times
        batch = RGAsrs.RgaNormPtotPlot(
            RgaFile, masses, Pres, TimeRange=Interval, plot='NO')
        batches.append(batch)
    slopebatches = []
    # 1st loop goes over each batch phase (using the lenght of batchstarts)
    for index1, el in enumerate(BatchStarts):
        slopeMasses = []
        slopeError = []
# The carrier gas should be the 1st listed in the mass list
#        ycarrier = batches[index1][:,1]
        #print (ycarrier.mean())
        for index2, el in enumerate(masses[1:], start=2):
            # start = 2 because on the batches matrix the column 0 is the time col 1 is the carrier gas
            # start from mass 1 because because mass 0 is the carrier gas

            x = batches[index1][:, 0]
            y = batches[index1][:, index2]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x, y)
            # plt.plot(x,y,'-p')    # uncomment this line to see the fit
            # yfit=x*slope+intercept    # uncomment this line to see the fit
            # plt.plot(x,yfit)          # uncomment this line to see the fit
            slopeMasses.append(slope)
            slopeError.append(std_err)
        if index1 == 0:
            slopebatches = slopeMasses
            if savefile != 'NO':
                line1 = '#   '
                FileSaveName = RgaFile[:-4]+'_2slopes.txt'
                FileSaveNameErr = RgaFile[:-4]+'_2slopesErr.txt'
                f = open(FileSaveName, 'w+')
                g = open(FileSaveNameErr, 'w+')
                for el in masses[1:]:
                    name = 'Mass '+str(el)
                    line1 = str(line1) + name + '   '
                f.writelines(line1 + '   ' + 'Time'+'\n')
                g.writelines(line1 + '   ' + 'Time'+'\n')
                f.writelines('   '.join('%5g' % e for e in slopeMasses) +
                             '   '+str(BatchStarts[index1]) + '\n')
                g.writelines('   '.join('%5g' % e for e in slopeError) +
                             '   '+str(BatchStarts[index1]) + '\n')
        else:
            #slopebatches = np.vstack([slopebatches,slopeMasses])
            if savefile != 'NO':
                f.writelines('   '.join('%5g' % e for e in slopeMasses) +
                             '   '+str(BatchStarts[index1]) + '\n')
                g.writelines('   '.join('%5g' % e for e in slopeError) +
                             '   '+str(BatchStarts[index1]) + '\n')
    if savefile != 'NO':
        f.close()
        g.close()

    return BatchStarts, slopebatches


def gasSlopesMapPar(Xcatfile, RgaFile, masses, shi, inte, TimeRange=None, savefile='NO', corrF_Ar=None, corrF_Flux=None):
    '''Load the masses e.g. [40,28,32,44] from RGA/XCAt files for the desired experiment and in the 
    selected time range it looks for the batch phases and fits the first 20 seconds of each batch phase 
    with a linear interpolation returning the slope for each gas-batch session.

    It also normalises the gas carrier that MUST be the FIRST MASS OF THE LIST (gas Ar often) 
    and report the partial pressures to the total pressure in the reactor. If it is not Ar check the 
    "gasScalingFactor" function.

    it assumes that during the batch phase the flux is not changed but simply goes to the drain.

    The resulta are: 
    Column 1 ==> mass 1 slopes
    Column 2 ==> mass 2 slopes

    slopes are in bar/sec

    shift and inter must be arrays eg: [6,16],[5,20]
    It searches the max slope in the shift= 5==>20 secs from the flow stop
    considering vectors of datapoints interval 5 ==> 25 pts   '''

    shifts = np.arange(shi[0], shi[1])
    intervals = np.arange(inte[0], inte[1])
    SlopesInt = []
    SlopesShi = []
    for shi in shifts:
        #batchTime,slopes = gasSlopes(directory+'20160212-215722.txt',directory+'NH3O2Part_Tscan_20160212.txt',[40,2,12,15,17,18,28,30,32,44],TimeRange=[starttime,deltaTime],savefile = 'YES', corrF_Ar = 9, corrF_Flux = 2.19)
        for interv in intervals:
            batchTime, slopes = gasSlopes(Xcatfile, RgaFile, masses, TimeRange=None, savefile='NO',
                                          shift=shi, inter=interv, corrF_Ar=corrF_Ar, corrF_Flux=corrF_Flux)
            SlopesInt = np.stack((SlopesInt, slopes), axis=2)
        SlopesShi = np.stack((SlopesShi, SlopesInt), axis=3)
    return SlopesShi


def gasFlow(Xcatfile, RgaFile, masses, shift=-20, inter=15, TimeRange=None, savefile='NO', corrF_Ar=None, corrF_Flux=None):
    '''Load the masses e.g. [40,28,32,44] from RGA/XCAt files for the desired experiment and in the 
    selected time range it looks for the batch phases and exctract the 20 seconds before each batch phase 
    and average the value.

    It also normalises the gas carrier that MUST be the FIRST MASS OF THE LIST (gas Ar often) 

    It reports the partial pressures to the total pressure in the reactor. If it is not Ar check the 
    "gasScalingFactor" function.

    it assumes that during the batch phase the flux is not changed but simply goes to the drain.

    The resulta are: 
    Column 1 ==> mass 1 averaged value of the 20 sec flow
    Column 2 ==> mass 2 averaged value of the 20 sec flow

    shift must be negative to look before the batch,(usually in macro we used flow==>batch)
    inter must be smaller or equal then shift

    output is in bar'''

    BatchStarts = getSection(
        Xcatfile, mixPos=3, mrsPos=10, TimeRange=TimeRange)
    BatchStarts = BatchStarts - TimeRange[0]
    print(BatchStarts)
    ScaleFactors = gasScalingFactor(
        Xcatfile, BatchStarts, TimeRange=TimeRange, corrF_Ar=corrF_Ar, corrF_Flux=corrF_Flux)
    #print (ScaleFactors)
    flows = []
    for el in BatchStarts:
        el = el + TimeRange[0]
        # define the interval to fit: [starting point, delta]
        Interval = [el+shift, inter]
#        print(Interval)
        # ==> computationally poor step, I have to read the file n times
        flow = RGAsrs.RgaSelect(RgaFile, masses, TimeRange=Interval)
#        print (np.shape(batch))
        # flows is a list of matrices, each matrix is a time
        flows.append(flow)
        # column and RGA data referring to a desired mass at the beginning of the (batch) phase
#    here we go in for the slope for each gas.
#    return batches

    aveflows = []
    # 1st loop goes over each batch phase (using the lenght of batchstarts)
    for index1, el in enumerate(BatchStarts):
        aveMasses = []
        aveError = []
# The carrier gas should be the 1st listed in the mass list
        ycarrier = flows[index1][:, 1]
        #print (ycarrier.mean())
        for index2, el in enumerate(masses[1:], start=2):
            # start = 2 because on the batches matrix the column 0 is the time col 1 is the carrier gas
            # start from mass 1 because because mass 0 is the carrier gas
            y = flows[index1][:, index2]
            ynorm = (y/ycarrier)*ScaleFactors[index1]
            yave = ynorm.mean()
            aveMasses.append(yave)
            yaves = np.ones(np.shape(ynorm)[0])*yave
            std_err = mse(yaves, ynorm)
            aveError.append(std_err)
        if index1 == 0:
            aveflows = aveMasses
            if savefile != 'NO':
                line1 = '#   '
                FileSaveName = RgaFile[:-4]+'_aveFlows.txt'
                FileSaveNameErr = RgaFile[:-4]+'_aveErr.txt'
                f = open(FileSaveName, 'w+')
                g = open(FileSaveNameErr, 'w+')
                for el in masses[1:]:
                    name = 'Mass '+str(el)
                    line1 = str(line1) + name + '   '
                f.writelines(line1+'\n')
                g.writelines(line1+'\n')
                f.writelines('   '.join('%5g' % e for e in aveMasses) + '\n')
                g.writelines('   '.join('%5g' % e for e in aveError) + '\n')
        else:
            aveflows = np.vstack([aveflows, aveMasses])
            if savefile != 'NO':
                f.writelines('   '.join('%5g' % e for e in aveMasses) + '\n')
                g.writelines('   '.join('%5g' % e for e in aveError) + '\n')
    f.close()
    g.close()

    return BatchStarts, aveflows


def gasFlow2(Xcatfile, RgaFile, masses, Pres, shift=-20, inter=15, TimeRange=None, savefile='NO', corrF_Ar=None, corrF_Flux=None):
    '''Load the masses e.g. [40,28,32,44] from RGA/XCAt files for the desired experiment and in the 
    selected time range it looks for the batch phases and exctract the 20 seconds before each batch phase 
    and average the value.

    it normalises to the Ptot in the reactor "Pres" == to the sum of the masses

    It reports the partial pressures to the total pressure in the reactor. If it is not Ar check the 
    "gasScalingFactor" function.

    it assumes that during the batch phase the flux is not changed but simply goes to the drain.

    The resulta are: 
    Column 1 ==> mass 1 averaged value of the 20 sec flow
    Column 2 ==> mass 2 averaged value of the 20 sec flow

    shift must be negative to look before the batch,(usually in macro we used flow==>batch)
    inter must be smaller or equal then shift

    output is in bar'''

    BatchStarts = getSection(
        Xcatfile, mixPos=3, mrsPos=10, TimeRange=TimeRange)
    BatchStarts = BatchStarts - TimeRange[0]
    print(BatchStarts)
    flows = []
    for el in BatchStarts:
        el = el + TimeRange[0]
        # define the interval to fit: [starting point, delta]
        Interval = [el+shift, inter]
        # ==> computationally poor step, I have to read the file n times
        flow = RGAsrs.RgaNormPtotPlot(
            RgaFile, masses, Pres, TimeRange=Interval, plot=None)
        # flows is a list of matrices, each matrix is a time
        flows.append(flow)
        # column and RGA data referring to a desired mass at the beginning of the (batch) phase
#    here we go in for the slope for each gas.
#    return batches

    aveflows = []
    # 1st loop goes over each batch phase (using the lenght of batchstarts)
    for index1, el in enumerate(BatchStarts):
        aveMasses = []
        aveError = []
        for index2, el in enumerate(masses[1:], start=2):
            # start = 2 because on the batches matrix the column 0 is the time col 1 is the carrier gas
            # start from mass 1 because because mass 0 is the carrier gas

            y = flows[index1][:, index2]
            yave = y.mean()
            aveMasses.append(yave)
            yaves = np.ones(np.shape(y)[0])*yave
            std_err = mse(yaves, y)
            aveError.append(std_err)
        if index1 == 0:
            aveflows = aveMasses
            if savefile != 'NO':
                line1 = '#   '
                FileSaveName = RgaFile[:-4]+'_2aveFlows.txt'
                FileSaveNameErr = RgaFile[:-4]+'_2aveErr.txt'
                f = open(FileSaveName, 'w+')
                g = open(FileSaveNameErr, 'w+')
                for el in masses[1:]:
                    name = 'Mass '+str(el)
                    line1 = str(line1) + name + '   '
                f.writelines(line1+'\n')
                g.writelines(line1+'\n')
                f.writelines('   '.join('%5g' % e for e in aveMasses) + '\n')
                g.writelines('   '.join('%5g' % e for e in aveError) + '\n')
        else:
            aveflows = np.vstack([aveflows, aveMasses])
            if savefile != 'NO':
                f.writelines('   '.join('%5g' % e for e in aveMasses) + '\n')
                g.writelines('   '.join('%5g' % e for e in aveError) + '\n')
    f.close()
    g.close()
    return BatchStarts, aveflows


def slope2mol(slopefile, temp, tempk=None):
    '''Meant to read the output file from the gasSlope function and transform the 
    slopes there in (bar/sec) into (molecules/sec) for a given temperature (Kelvin).
    Assuming the gasses behaving as ideal gasses
    It saves the file as slopefile[:-4]+'_molsec.txt '''

    data = np.genfromtxt(slopefile, dtype='float', comments='#', delimiter='')
    elaps = data[:, -1]
    data = data[0:, 0:-1]
    fo = open(slopefile, "r")
    line1 = fo.readline()
    fo.close()
    R = 8.314e-2  # L Bar / K Mol
    Vol = 0.01  # Reactor volume in liters
    avog = 6.023e23  # molecules/Mol
    temp = np.asanyarray(temp)
    if np.size(temp) > 1:  # using a temperature vector to recalculate the number of molecules
        Tc = 'Tramp'
        dataout = np.empty(len(data[0])+2)
        for index, el in enumerate(temp):
         #           for ind, el in enumerate(masslist[1:], start=2):
            if tempk == None:
                dataLout = ((data[index]*Vol)/(R*temp[index]))*avog
                dataLout = np.append(dataLout, temp[index])
                dataLout = np.append(dataLout, elaps[index])
                dataout = np.c_[dataout, dataLout]
            if tempk != None:
                dataLout = ((data[index]*Vol)/(R*tempk))*avog
                dataLout = np.append(dataLout, temp[index])
                dataLout = np.append(dataLout, elaps[index])
                dataout = np.c_[dataout, dataLout]

#        print (dataout.shape)

        dataout = dataout[:, 1:]
        #print('dataout shape', dataout.shape)
        FileSaveName = slopefile[:-4]+'_'+str(Tc)+'K_molsec.txt'
        fo = open(FileSaveName, 'w+')
        line1 = line1[:-2]+'   Temp'+'    Time\n'
        fo.writelines(line1)
        for i in range(np.shape(data)[0]):
            matline = (dataout[:, i])
            fo.writelines('   '.join('%5g' % e for e in matline) + '\n')
        fo.close()

    if np.size(temp) <= 1:  # using temperature constant to recalculate the number of molecules
        Tc = temp
        dataout = ((data*Vol)/(R*temp))*avog
        FileSaveName = slopefile[:-4]+'_'+str(Tc)+'K_molsec.txt'
        fo = open(FileSaveName, 'w+')
        fo.writelines(line1)
        for i in range(np.shape(data)[0]):
            matline = np.append(dataout[i], elaps[i])
            fo.writelines('   '.join('%5g' % e for e in matline) + '\n')
        fo.close()
    return dataout


def flow2mol(flowfile, temp, p0CO, p0O2,):
    '''For now is specifically tailored for the termramp of Venky's data
    meant to read the flowfile output of gasFlow function
    p0CO is the CO initial pressure readen by the RGA when the catalyst is NOT active
    p0O2 is the O2 initial pressure readen by the RGA when the catalyst is NOT active

    output of the function are the molecule/sec prodused or consumed per second'''
    data = np.genfromtxt(flowfile, dtype='float', comments='#', delimiter='')

    fo = open(flowfile, "r")
    line1 = fo.readline()
    fo.close()
    R = 8.314e-2  # L Bar / K Mol
    Vol = 0.01  # Reactor volume in liters
    avog = 6.023e23  # molecules/Mol
# consumption rate = (FlowOut - Flow In)/V
# ==> (Preactor - Pin)/(Pin / flowIn) ==> flow variation
    # * .005)/Vol #( Preactor - Pinitial / Pinitial )*entrance Flow ==> consumed flow/Volume for normalisation
    data[:, 0] = ((data[:, 0]-p0CO)/(p0CO/0.005))/Vol
    data[:, 1] = ((data[:, 1]-p0O2)/(p0O2/0.05))/Vol  # *.05)/Vol
    data[:, 2] = ((data[:, 2]-0.00399)/(0.00263/0.000588))/Vol
    # (0.003/0.00053) are the max pCO2 produced divided by its equivalent recalculated flux from CO consumption
# data now is in l/min      0.00273 is the bkg value for CO2 which is not constant across the all interval
    data[:, 0] = data[:, 0]/60
    data[:, 1] = data[:, 1]/60
    data[:, 2] = data[:, 2]/60
# data now is in l/sec
    temp = np.asanyarray(temp)
    dataout = np.empty([data.shape[1]])
    if np.size(temp) > 1:  # using a temperature vector to recalculate the number of molecules
        Tc = 'Tramp'
        dataout = np.empty(len(data[0])+1)
        for index, el in enumerate(temp):
            #           for ind, el in enumerate(masslist[1:], start=2):
            dataLout = ((data[index]*Vol)/(R*temp[index]))*avog
            dataLout = np.append(dataLout, temp[index])
            dataout = np.c_[dataout, dataLout]
#        print (dataout.shape)

        dataout = dataout[:, 1:]
        print('dataout shape', dataout.shape)
        FileSaveName = flowfile[:-4]+'_'+str(Tc)+'K_molsec.txt'
        fo = open(FileSaveName, 'w+')
        line1 = line1[:-2]+'   Temp\n'
        fo.writelines(line1)
        for i in range(np.shape(data)[0]):
            matline = (dataout[:, i])
            fo.writelines('   '.join('%5g' % e for e in matline) + '\n')
        fo.close()

    if np.size(temp) == 1:  # using a constant temperature to recalculate the number of molecules
        Tc = temp
        dataout = ((data*Vol)/(R*temp))*avog
        FileSaveName = flowfile[:-4]+'_'+str(Tc)+'K_molsec.txt'
        fo = open(FileSaveName, 'w+')
        fo.writelines(line1)
        for i in range(np.shape(data)[0]):
            matline = (dataout[i])
            fo.writelines('   '.join('%5g' % e for e in matline) + '\n')
        fo.close()
    return dataout


def LeakFactorPlot(filenameXCAT, filenameRGA, masslist, channelList, TimeRange=None):
    '''Try to check the behaviour of the leack factor across the experiment, 
    masslist anch channel list must refer to the gas fragment through that flow controller.
    eg: 2, 32,28 ==> H2, O2.CO
    '''
    GasLog = selectGas(filenameXCAT, TimeRange=TimeRange)

    channelpos = list()
    channels = ('NO', 'H2', 'O2', 'CO', 'Ar', 'val', 'shu', 'rea', 'dra')
    colors = ('-c', '-r', '-g', '-k', 'm', '-b', '-k', '-b')
    for el in channelList:
        if np.isreal(channels.index(el)):
            channelpos.append(channels.index(el))
        else:
            print('At least one channel is not contained in the data names')

    # plt.subplot(711)  #### Plot 1
    '''1) Flow measures
    '''
    flows = GasLog[el][:, 0]
    # flows=np.array([])
    for el in channelpos:
        #plt.plot(GasLog[el][:,0], GasLog[el][:,0], colors[el], label = channels[el])
        flows = np.c_([flows, GasLog[el][:, 0]])

    RgaGas = RGAsrs.RgaSelect(filenameRGA, masslist, TimeRange=TimeRange)

    rgas = np.array([])

    for ind, el in enumerate(masslist[1:], start=0):

       # plt.plot(RgaGas[:,0], RgaGas[:,ind], colors[ind-1])
        rgas = np.c_([rgas, RgaGas[:, ind]])

    if np.shape(rgas)[1] == np.shape(flows)[1]:
        timest = min(rgas[:, 0])
        timend = max(rgas[:, 0])
        tt = np.arange(timest, timend, 1)
        for el in np.range(np.shape(rgas)[1]):
            rga = np.interp(tt, rgas[:, 0], rgas[:, el])
            flow = np.interp(tt, flows[:, 0], flows[:, el])
            plt.plot(tt, rga/flow)


#    pl.xlim(0,TimeRange[1])
#    pl.ylim(yy.min(), yy.max())
#    pl.ylabel('Normalised Int (A.U.)',fontsize=16)
#    pl.xlabel('Time (s)', fontsize=16)
#    pl.legend(loc=1)
#    plt.show()
