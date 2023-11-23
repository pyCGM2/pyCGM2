"""
This Module contains the pyCGM2 plot filter `PlottingFilter`.

This  `PlottingFilter` requires a  `PlotViewer`, then displays the plot panel

"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pyCGM2
from pyCGM2.Report import plot
from pyCGM2.Report.Viewers.plotViewers import PlotViewer

from typing import List, Tuple, Dict, Optional, Union, Callable

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


    def setExport(self,path:str,filename:str,format:str):
        '''Set filename of the export file

        Args:
            path (str): folder path
            filename (str): filename
            format (str) : file format
        '''
        self.m_path = path
        self.m_fileName = filename
        self.m_format = format



    def setViewer(self,concretePlotViewer:PlotViewer):
        """set the plot viewer

        Args:
            concretePlotViewer (PlotViewer): a plot viewer

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

    def setYlimits(self, axisIndex:int, min:float, max:float):
        """set the y-axis boundaries

        Args:
            axisIndex (int): index of the `matplotlib.figure.axis` instance
            min (float): minimum value
            max (float): maximum value

        """

        self.__concretePlotViewer.fig.axes[axisIndex].set_ylim([min,max])

    def setHorizontalLines(self, idict:Dict):
        """set horizontal lines from a dict whom key is the axis title 

        Args:
            idict (dict): dictionnary with axis label as key
        """
        
        for key in idict:
            for axisIt in self.__concretePlotViewer.fig.axes:
                if axisIt.get_title() in key:
                    for it in idict[key]:
                        value=it[0]
                        color=it[1]
                        axisIt.axhline(value,color=color,ls='dashed')
                    break



    def setAutomaticYlimits(self):
        """set Y-axis boundaries to default values

        """
        self.__concretePlotViewer.setAutomaticYlimits(True)

    def setTitle(self,title:str):
        """Set the plot panel title

        Args:
            title (str): title

        """
        self.m_title=title

    def displaySignificantDiffererence(self,axisIndex:int,clusters:List):
        """display sgnificant frames

        Args:
            axisIndex (int): index of the plt.Axes
            clusters (list): cluster of significant frames 
        """

        plot.addRectanglePatches(self.__concretePlotViewer.fig.axes[axisIndex],clusters)
