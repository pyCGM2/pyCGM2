from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import btk

import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional, Union

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

class AnomalyCorrectionProcedure(object):
    """Base class for implementing anomaly correction procedures.

    This class serves as a foundation for developing specific anomaly correction techniques.
    It can be extended to add specific initialization and methods for handling different types of anomalies.
    """
    def __init__(self):
        pass


class MarkerAnomalyCorrectionProcedure(AnomalyCorrectionProcedure):
    """Subclass of AnomalyCorrectionProcedure for correcting anomalies in marker data.

    This class implements a procedure to identify and correct anomalies in marker trajectories
    using various parameters and methods.

    Attributes:
        m_markers (Union[List[str], str]): List of marker labels or a single marker label.
        m_anomalyIndexes (List[int]): Indices of detected anomalies in the marker data.
        _plot (bool): Flag to indicate if the plot should be displayed. Defaults to False.
        _distance_threshold (int): Threshold distance for clustering anomalies. Defaults to 10.
    """
    


    def __init__(self, markers:Union[List,str], anomalyIndexes:List[int], plot:bool=False, **kwargs):
        """
        Initialize the MarkerAnomalyCorrectionProcedure class with given parameters.

        Args:
            markers (Union[List[str], str]): List of marker labels or a single marker label.
            anomalyIndexes (List[int]): List of indices where anomalies are detected in the marker data.
            plot (bool, optional): Flag to indicate if the plot should be displayed. Defaults to False.
        """
        super(MarkerAnomalyCorrectionProcedure,self).__init__()
        
    

        if type(markers) == str:
            markers = [markers]

        self.m_markers = markers
        self.m_anomalyIndexes = anomalyIndexes
        self._plot = plot

        self._distance_threshold = 10 if "distance_threshold" not in kwargs else kwargs[
            "distance_threshold"]

    def run(self, acq:btk.btkAcquisition, filename:str):
        """
        Execute the anomaly correction procedure on the given acquisition data.

        Args:
            acq (btk.btkAcquisition): An instance of a btk acquisition object.
            filename (str): The filename of the data to be processed.

        Returns:
            btk.btkAcquisition: The acquisition object after applying anomaly corrections.
        """

        ff = acq.GetFirstFrame()

        for marker in self.m_markers:

            if marker in self.m_anomalyIndexes and self.m_anomalyIndexes[marker] != []:
                indices_frameMatched = self.m_anomalyIndexes[marker]
                indices = [it-ff for it in indices_frameMatched]

                pointValues = acq.GetPoint(marker).GetValues()
                values = np.linalg.norm(pointValues, axis=1)
                values0 = np.linalg.norm(pointValues, axis=1)
                residualValues = acq.GetPoint(marker).GetResiduals()

                if len(indices) > 1:
                    clustering_model = AgglomerativeClustering(distance_threshold=self._distance_threshold, n_clusters=None).fit(
                        np.array(indices).reshape((len(indices), 1)))
                    n_clusters = clustering_model.n_clusters_

                    for i in range(0, n_clusters):
                        beg = indices[np.where(
                            clustering_model.labels_ == i)[0][0]]
                        end = indices[np.where(
                            clustering_model.labels_ == i)[0][-1]]
                        LOGGER.logger.warning(
                            "[pycgm2] correction from %i to %i" % (beg, end))
                        values[beg:end+1] = np.nan
                        pointValues[beg:end+1] = 0
                        residualValues[beg:end+1] = -1.0

                else:
                    beg = indices[0]-1
                    end = indices[0]+1
                    LOGGER.logger.warning(
                        "[pycgm2] correction from %i to %i" % (beg, end))
                    values[beg:end+1] = np.nan
                    pointValues[beg:end+1] = 0
                    residualValues[beg:end+1] = -1.0

                acq.GetPoint(marker).SetResiduals(residualValues)
                acq.GetPoint(marker).SetValues(pointValues)

                # if self._plot:
                #     fig, axs = plt.subplots(1)
                #     fig.suptitle('trajectory of marker %s'%(marker))
                #     axs.plot(values0)
                #     axs.plot(values,"-r")
                #     # axs.set_ylim([2040,2100])
                #     plt.show()
            else:
                LOGGER.logger.debug(
                    "[pyCGM2] -  No anomalies detected for marker %s" % marker)

        return acq
