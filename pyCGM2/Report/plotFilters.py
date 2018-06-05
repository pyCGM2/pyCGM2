# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# pyCGM2
#import pyCGM2
import pyCGM2.Processing.analysis as CGM2analysis

# openMA
import ma.io
import ma.body

# ------ FILTER -----------
class PlottingFilter(object):
    """
        **Description :** This filter calls a concrete PlotBuilder and run  its plot-embedded method
    """

    def __init__(self):

        self.__concretePlotViewer = None
        self.m_path = None
        self.m_pdfName = None

    def setPath(self,path):
        '''
            **Description :** define path  of the desired output folder

            :Parameters:
             - `path` (str) - path must end with \\


        '''

        self.m_path = path

    def setPdfName(self,name):
        '''
            **Description :** set filename of the pdf

            :Parameters:
             - `name` (str)


        '''

        self.m_pdfName = name



    def setViewer(self,concretePlotViewer):
        '''
            **Description :** load a concrete plot builder

            :Parameters:
             - `concretePlotViewer` (pyCGM2.Report.plot PlotBuilder) - concrete plot builder from pyCGM2.Report.plot module

        '''

        self.__concretePlotViewer = concretePlotViewer

    def plot(self):
        '''
            **Description :** Generate plot panels

        '''

        #self.__concretePlotViewer.plotPanel(self.m_path,self.m_pdfName)
        self.fig = self.__concretePlotViewer.plotPanel()


        if self.m_path is not None and self.m_pdfName is not None:
            pp = PdfPages(str(self.m_path+ self.m_pdfName+".pdf"))
            pp.savefig(self.fig)
            pp.close()

    def setYlimits(self, axisIndex, min, max):
        self.__concretePlotViewer.fig.axes[axisIndex].set_ylim([min,max])

    def setHorizontalLine(self, axisIndex, value,color= "black"):
        self.__concretePlotViewer.fig.axes[axisIndex].axhline(value,color=color,ls='dashed')
