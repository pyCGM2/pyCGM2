import ipdb
import logging


import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2.Report import plot,plotFilters,plotViewers
from pyCGM2.Tools import trialTools

class PlotTest():


    @classmethod
    def temporalKinematicPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]


        trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames[0])

        # # viewer
        kv = plotViewers.TemporalGaitKinematicsPlotViewer(trial)
        #
        # # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()

    @classmethod
    def temporalKineticPlotPanel(cls):

        # ----DATA-----
        DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\plot\\gaitPlot\\"
        modelledFilenames = ["gait Trial 03 - viconName.c3d"]

        trial =trialTools.smartTrialReader(DATA_PATH,modelledFilenames[0])
        # viewer
        kv = plotViewers.TemporalGaitKineticsPlotViewer(trial)

        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        pf.plot()

        plt.show()

if __name__ == "__main__":

    plt.close("all")

    PlotTest.temporalKineticPlotPanel()
