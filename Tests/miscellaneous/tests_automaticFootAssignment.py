# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 15:14:18 2016

@author: aaa34169
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pdb

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# btk
pyCGM2.CONFIG.addBtk()
import btk

# pyCGM2
from pyCGM2.Tools import  btkTools,trialTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Processing import cycle,analysis


pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body




class tests():

    @classmethod
    def twoPF_FP1none_FP2none(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\forceplates\\FootAssignementAutoamticGeneralEvent\\"

        # --- Motion 1
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 13.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        forceplates.addForcePlateGeneralEvents(acqGait,"XX")

        btkTools.smartWriter(acqGait,str(MAIN_PATH+"twoPF_FP1none_FP2none.c3d"))


        modelledFilenames = ["twoPF_FP1none_FP2none.c3d"]

        #---- GAIT CYCLES FILTER PRELIMARIES
        #--------------------------------------------------------------------------
        # distinguishing trials for kinematic and kinetic processing

        # - kinematic Trials
        kinematicTrials=[]
        kinematicFilenames =[]
        for kinematicFilename in modelledFilenames:
            kinematicFileNode = ma.io.read(str(MAIN_PATH+kinematicFilename))
            kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
            trialTools.sortedEvents(kinematicTrial)

            longitudinalAxis,forwardProgression,globalFrame = trialTools.findProgression(kinematicTrial,"LHEE")

            kinematicTrials.append(kinematicTrial)
            kinematicFilenames.append(kinematicFilename)

        # - kinetic Trials ( check if kinetic events)
        kineticTrials,kineticFilenames,flag_kinetics =  trialTools.automaticKineticDetection(MAIN_PATH,modelledFilenames)

        #---- GAIT CYCLES FILTER
        #--------------------------------------------------------------------------
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=kinematicTrials,
                                               kinematicTrials = kinematicTrials,
                                               kineticTrials = kineticTrials,
                                               emgTrials=None)

        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()

        # TESTING
        np.testing.assert_equal(cycles.kineticCycles is None, True)
        np.testing.assert_equal(cycles.kineticCycles, None)



    @classmethod
    def twoPF_FP1left_FP2none(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\forceplates\\FootAssignementAutoamticGeneralEvent\\"

        # --- Motion 1
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 13.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        forceplates.addForcePlateGeneralEvents(acqGait,"LX")

        btkTools.smartWriter(acqGait,str(MAIN_PATH+"twoPF_FP1left_FP2none.c3d"))


        modelledFilenames = ["twoPF_FP1left_FP2none.c3d"]

        #---- GAIT CYCLES FILTER PRELIMARIES
        #--------------------------------------------------------------------------
        # distinguishing trials for kinematic and kinetic processing

        # - kinematic Trials
        kinematicTrials=[]
        kinematicFilenames =[]
        for kinematicFilename in modelledFilenames:
            kinematicFileNode = ma.io.read(str(MAIN_PATH+kinematicFilename))
            kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
            trialTools.sortedEvents(kinematicTrial)

            longitudinalAxis,forwardProgression,globalFrame = trialTools.findProgression(kinematicTrial,"LHEE")

            kinematicTrials.append(kinematicTrial)
            kinematicFilenames.append(kinematicFilename)

        # - kinetic Trials ( check if kinetic events)
        kineticTrials,kineticFilenames,flag_kinetics =  trialTools.automaticKineticDetection(MAIN_PATH,modelledFilenames)

        #---- GAIT CYCLES FILTER
        #--------------------------------------------------------------------------
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=kinematicTrials,
                                               kinematicTrials = kinematicTrials,
                                               kineticTrials = kineticTrials,
                                               emgTrials=None)

        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()

        # TESTING

        np.testing.assert_equal(len(cycles.kineticCycles), 1)
        np.testing.assert_equal(cycles.kineticCycles[0].context, "Left")
        np.testing.assert_equal(cycles.kineticCycles[0].begin, 253)


    @classmethod
    def twoPF_FP1left_FP2right(cls):

        MAIN_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "operations\\forceplates\\FootAssignementAutoamticGeneralEvent\\"

        # --- Motion 1
        gaitFilename="MRI-US-01, 2008-08-08, 3DGA 13.c3d"
        acqGait = btkTools.smartReader(str(MAIN_PATH +  gaitFilename))
        #forceplates.appendForcePlateCornerAsMarker(acqGait)
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
        forceplates.addForcePlateGeneralEvents(acqGait,"LR")

        btkTools.smartWriter(acqGait,str(MAIN_PATH+"twoPF_FP1left_FP2right.c3d"))


        modelledFilenames = ["twoPF_FP1left_FP2right.c3d"]

        #---- GAIT CYCLES FILTER PRELIMARIES
        #--------------------------------------------------------------------------
        # distinguishing trials for kinematic and kinetic processing

        # - kinematic Trials
        kinematicTrials=[]
        kinematicFilenames =[]
        for kinematicFilename in modelledFilenames:
            kinematicFileNode = ma.io.read(str(MAIN_PATH+kinematicFilename))
            kinematicTrial = kinematicFileNode.findChild(ma.T_Trial)
            trialTools.sortedEvents(kinematicTrial)

            longitudinalAxis,forwardProgression,globalFrame = trialTools.findProgression(kinematicTrial,"LHEE")

            kinematicTrials.append(kinematicTrial)
            kinematicFilenames.append(kinematicFilename)

        # - kinetic Trials ( check if kinetic events)
        kineticTrials,kineticFilenames,flag_kinetics =  trialTools.automaticKineticDetection(MAIN_PATH,modelledFilenames)

        #---- GAIT CYCLES FILTER
        #--------------------------------------------------------------------------
        cycleBuilder = cycle.GaitCyclesBuilder(spatioTemporalTrials=kinematicTrials,
                                               kinematicTrials = kinematicTrials,
                                               kineticTrials = kineticTrials,
                                               emgTrials=None)

        cyclefilter = cycle.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycles = cyclefilter.build()

        # TESTING

        np.testing.assert_equal(len(cycles.kineticCycles), 2)
        np.testing.assert_equal(cycles.kineticCycles[0].context, "Left")
        np.testing.assert_equal(cycles.kineticCycles[0].begin, 253)
        np.testing.assert_equal(cycles.kineticCycles[1].context, "Right")
        np.testing.assert_equal(cycles.kineticCycles[1].begin, 201)


if __name__ == "__main__":
    plt.close("all")

    tests.twoPF_FP1none_FP2none()
    tests.twoPF_FP1left_FP2none()
    tests.twoPF_FP1left_FP2right()
