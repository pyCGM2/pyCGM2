from pyCGM2.Tools import btkTools
from pyCGM2.Signal import anomaly

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import logging
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

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


class MarkerAnomalyCorrectionProcedure(object):
    def __init__(self,markers,anomalyIndexes,plot=False,**options):

        if type(markers) == str:
            markers = [markers]

        self.m_markers = markers
        self.m_anomalyIndexes = anomalyIndexes
        self._plot = plot

        self._distance_threshold = 10 if "distance_threshold" not in options else options["distance_threshold"]


    def run(self,acq,filename):

        ff = acq.GetFirstFrame()

        for marker in self.m_markers:

            indices_frameMatched = self.m_anomalyIndexes[marker]
            indices = [it-ff for it in indices_frameMatched]

            clustering_model = AgglomerativeClustering(distance_threshold=self._distance_threshold, n_clusters=None).fit(np.array(indices).reshape((len(indices),1)))
            n_clusters = clustering_model.n_clusters_

            # plt.title('Hierarchical Clustering Dendrogram')
            # # plot the top three levels of the dendrogram
            # plot_dendrogram(clustering, truncate_mode='level', p=3)
            # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            # plt.show()


            pointValues = acq.GetPoint(marker).GetValues()
            values = np.linalg.norm(pointValues,axis=1)
            values0 = np.linalg.norm(pointValues,axis=1)

            residualValues = acq.GetPoint(marker).GetResiduals()


            for i in range(0, n_clusters):
                beg = indices[np.where(clustering_model.labels_==i)[0][0]]
                end = indices[np.where(clustering_model.labels_==i)[0][-1]]
                logging.warning("[pycgm2] correction from %i to %i"%(beg,end))
                values[beg:end+1]= np.nan
                residualValues[beg:end+1] = -1.0

            acq.GetPoint(marker).SetResiduals(residualValues)
            acq.GetPoint(marker).SetValues(pointValues)

            if self._plot:
                fig, axs = plt.subplots(1)
                fig.suptitle('trajectory of marker %s'%(marker))
                axs.plot(values0)
                axs.plot(values,"-r")
                # axs.set_ylim([2040,2100])
                plt.show()

        return acq
