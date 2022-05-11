# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.CGM2 import  cgm
from pyCGM2.Tools import btkTools


class RepresentativeProcedure(object):
    def __init__(self):
        pass



class Sangeux2015Procedure(RepresentativeProcedure):
    """Representative detection method of Sangeux 2015

    **reference**

    Sangeux, Morgan A simple method to choose the most representative stride and detect outliers.

    """

    def __init__(self):
        super(Sangeux2015Procedure, self).__init__()

        self.m_data = dict()
        self.m_data["Left"] = []
        self.m_data["Right"] = []


    def setDefaultData(self):
        """set default data according Sangeux 2015
        """

        self.setData('Left',"LPelvisAngles",[0,1,2])
        self.setData('Left',"LHipAngles",[0,1,2])
        self.setData('Left',"LKneeAngles",[0,1])
        self.setData('Left',"LAnkleAngles",[0])
        self.setData('Left',"LFootProgressAngles",[2])

        self.setData('Right',"RPelvisAngles",[0,1,2])
        self.setData('Right',"RHipAngles",[0,1,2])
        self.setData('Right',"RKneeAngles",[0,1])
        self.setData('Right',"RAnkleAngles",[0])
        self.setData('Right',"RFootProgressAngles",[2])


    def setData(self,EventContext,Label,indexes):
        """populate data

        Args:
            EventContext (str): event context
            Label (str): kinematic model output label
            indexes (list): axis indexes

        ```python
           proc = Sangeux2015Procedure
           proc.setData("Left","LHipAngles",[0,2]) # 0:flexion and 2:transverse rotation
        ```

        """
        self.m_data[EventContext].append([Label,indexes])


    def _calculateFmd(self,medianeValues,values):
        return np.divide( np.sum( np.abs(values[1:100]-medianeValues[1:100]))+
                    0.5 *( np.abs(values[0]-medianeValues[0]) +
                            np.abs(values[100]-medianeValues[100])), 100)


    def _run(self, analysis):

        out=dict()
        for eventContext in self.m_data:

            fmds=[]
            for data in self.m_data[eventContext]:
                label = data[0]
                axes  = data[1]

                mediane = analysis.kinematicStats.data[label,eventContext]["median"]
                stridesValues = analysis.kinematicStats.data[label,eventContext]["values"]
                nStrides = len(stridesValues)

                for axis in axes:
                    fmd_byStride = [label,axis]
                    for strideIndex in range(0,nStrides):
                        fmd_byStride.append(self._calculateFmd(mediane[:,axis],stridesValues[strideIndex][:,axis]))
                    fmds.append(fmd_byStride)


            colnames = ['Label', 'Axis'] +[str(i) for i in range(0,nStrides)]
            df = pd.DataFrame(fmds, columns=colnames)

            sortedDf = df.iloc[:,2:].mean().sort_values()

            sortedStrideIndex =  [int(i) for i in sortedDf.index]

            out[eventContext] = sortedStrideIndex[0]

        return out
