# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Report
#APIDOC["Draft"]=False
#--end--

"""
This Module contains the pyCGM2 plot filter `PlottingFilter`.

This  `PlottingFilter` requires a  `PlotViewer`, then displays the plot panel

"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pyCGM2
from pyCGM2.Report import plot




# ------ FILTER -----------
class PlottingFilter(object):
    """The plot filter
    """

    def __init__(self):

        self.__concretePlotViewer = None
        self.m_path = None
        self.m_fileName = None
        self.m_format = None
        self.m_title = None


    def setExport(self,path,filename,format):
        '''Set filename of the export file

        Args:
            path (str): folder path
            filename (str): filename
            format (str) : file format
        '''
        self.m_path = path
        self.m_fileName = filename
        self.m_format = format



    def setViewer(self,concretePlotViewer):
        """set the plot viewer

        Args:
            concretePlotViewer (pyCGM2.report.(Viewers)): a plot viewer

        """

        self.__concretePlotViewer = concretePlotViewer

    def plot(self):
        """generate plot panel
        """


        #self.__concretePlotViewer.plotPanel(self.m_path,self.m_pdfName)
        self.fig = self.__concretePlotViewer.plotPanel()
        if self.m_title is not None: self.__concretePlotViewer.fig.suptitle(self.m_title)


        if self.m_path is not None and self.m_fileName is not None:
            if self.m_format is "pdf":
                pp = PdfPages((self.m_path+ self.m_fileName+".pdf"))
                pp.savefig(self.fig)
                pp.close()
            else:
                plt.savefig((self.m_path+ self.m_fileName+"."+self.m_format))

        return self.fig

    def setYlimits(self, axisIndex, min, max):
        """set the y-axis boundaries

        Args:
            axisIndex (int): index of the `matplotlib.figure.axis` instance
            min (double): minimum value
            max (double): maximum value

        """

        self.__concretePlotViewer.fig.axes[axisIndex].set_ylim([min,max])

    def setHorizontalLine(self, axisIndex, value,color= "black"):
        """set horizontal lines

        Args:
            axisIndex (int): index of the `matplotlib.figure.axis` instance
            value (double): y-axis value
            color (str,Optional[black]): line color
        """

        self.__concretePlotViewer.fig.axes[axisIndex].axhline(value,color=color,ls='dashed')


    def setAutomaticYlimits(self):
        """set Y-axis boundaries to default values

        """
        self.__concretePlotViewer.setAutomaticYlimits(True)

    def setTitle(self,title):
        """Set the plot panel title

        Args:
            title (str): title

        """
        self.m_title=title

    def displaySignificantDiffererence(self,axisIndex,clusters):

        plot.addRectanglePatches(self.__concretePlotViewer.fig.axes[axisIndex],clusters)
