# -*- coding: utf-8 -*-
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np

import pyCGM2
LOGGER = pyCGM2.LOGGER


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
    def __init__(self):
        pass


class MarkerAnomalyCorrectionProcedure(AnomalyCorrectionProcedure):
    """procedure to correct marker anomaly

    Args:
        markers (list): marker labels
        anomalyIndexes (list): indexes of the detected anomalies
        plot (bool): enable plot

    Keyword Arguments:
        distance_threshold (float): distance threshold between indexes


    """

    def __init__(self, markers, anomalyIndexes, plot=False, **kwargs):
        super(MarkerAnomalyCorrectionProcedure,self).__init__()

        if type(markers) == str:
            markers = [markers]

        self.m_markers = markers
        self.m_anomalyIndexes = anomalyIndexes
        self._plot = plot

        self._distance_threshold = 10 if "distance_threshold" not in kwargs else kwargs[
            "distance_threshold"]

    def run(self, acq, filename):
        """ run the procedure

        Args:
            acq (btk.Acquisition): a btk acquisition instantce
            filename (str): filename

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
