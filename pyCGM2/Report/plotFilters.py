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
        self.m_fileName = None
        self.m_format = None


    def setExport(self,path,filename,format):
        '''
            **Description :** set filename of the pdf

            :Parameters:
             - `format` (str) : image format


        '''
        self.m_path = path
        self.m_fileName = filename
        self.m_format = format



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


        if self.m_path is not None and self.m_fileName is not None:
            if self.m_format is "pdf":
                pp = PdfPages(str(self.m_path+ self.m_fileName+".pdf"))
                pp.savefig(self.fig)
                pp.close()
            else:
                plt.savefig(str(self.m_path+ self.m_fileName+"."+self.m_format))

    def setYlimits(self, axisIndex, min, max):
        self.__concretePlotViewer.fig.axes[axisIndex].set_ylim([min,max])

    def setHorizontalLine(self, axisIndex, value,color= "black"):
        self.__concretePlotViewer.fig.axes[axisIndex].axhline(value,color=color,ls='dashed')
