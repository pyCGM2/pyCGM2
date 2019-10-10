# -*- coding: utf-8 -*-
import logging
import pyCGM2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages





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
        self.m_title = None


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
        if self.m_title is not None: self.__concretePlotViewer.fig.suptitle(self.m_title)


        if self.m_path is not None and self.m_fileName is not None:
            if self.m_format is "pdf":
                pp = PdfPages((self.m_path+ self.m_fileName+".pdf").decode("utf-8").encode(pyCGM2.ENCODER))
                pp.savefig(self.fig)
                pp.close()
            else:
                plt.savefig((self.m_path+ self.m_fileName+"."+self.m_format).decode("utf-8").encode(pyCGM2.ENCODER))

    def setYlimits(self, axisIndex, min, max):
        self.__concretePlotViewer.fig.axes[axisIndex].set_ylim([min,max])

    def setHorizontalLine(self, axisIndex, value,color= "black"):
        self.__concretePlotViewer.fig.axes[axisIndex].axhline(value,color=color,ls='dashed')

    def setTitle(self,title):
        self.m_title=title
