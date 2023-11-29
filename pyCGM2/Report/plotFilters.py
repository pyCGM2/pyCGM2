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
    """Manages the creation and display of plots using a specified PlotViewer.

    This filter is responsible for generating and exporting plots. It allows
    customization of plot appearance, including setting titles, Y-axis limits,
    and adding horizontal lines or significant difference markers.

    Attributes:
        __concretePlotViewer (PlotViewer): A plot viewer used to generate the plot panel.
        m_path (str): Path where the plot will be saved.
        m_fileName (str): Name of the file to save the plot.
        m_format (str): Format of the saved plot file.
        m_title (str): Title of the plot panel.
    """

    def __init__(self):
        """Initializes the PlottingFilter with default attributes."""

        self.__concretePlotViewer = None
        self.m_path = None
        self.m_fileName = None
        self.m_format = None
        self.m_title = None


    def setExport(self,path:str,filename:str,format:str):
        """Configures the export settings for the plot.

        Args:
            path (str): Folder path where the plot will be saved.
            filename (str): Name of the file to save the plot.
            format (str): Format of the saved plot file (e.g., 'pdf', 'png').
        """
        self.m_path = path
        self.m_fileName = filename
        self.m_format = format



    def setViewer(self,concretePlotViewer:PlotViewer):
        """Sets the plot viewer to be used for generating the plot panel.

        Args:
            concretePlotViewer (PlotViewer): A plot viewer instance.
        """

        self.__concretePlotViewer = concretePlotViewer

    def plot(self):
        """Generates and optionally saves the plot panel based on the current settings.

        Returns:
            A matplotlib figure object containing the generated plot.
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
        """Sets the Y-axis boundaries for a specific plot axis.

        Args:
            axisIndex (int): Index of the matplotlib.figure.axis instance.
            min (float): Minimum value for the Y-axis.
            max (float): Maximum value for the Y-axis.
        """

        self.__concretePlotViewer.fig.axes[axisIndex].set_ylim([min,max])

    def setHorizontalLines(self, idict:Dict):
        """Sets horizontal lines on the plot axes based on the provided dictionary.

        Args:
            idict (Dict): Dictionary with axis labels as keys and lists of (value, color) pairs as values.
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
        """Sets the Y-axis boundaries to default values for all plot axes."""
        self.__concretePlotViewer.setAutomaticYlimits(True)

    def setTitle(self,title:str):
        """Sets the title for the plot panel.

        Args:
            title (str): Title to set for the plot panel.
        """
        self.m_title=title

    def displaySignificantDiffererence(self,axisIndex:int,clusters:List):
        """Displays markers for significant frames on a specified plot axis.

        Args:
            axisIndex (int): Index of the matplotlib.figure.axis instance.
            clusters (List): List of clusters of significant frames.
        """

        plot.addRectanglePatches(self.__concretePlotViewer.fig.axes[axisIndex],clusters)
