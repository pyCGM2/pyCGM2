# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--


import numpy as np
from pyCGM2.Math import numeric
import pyCGM2
LOGGER = pyCGM2.LOGGER


class ScoreProcedure(object):
    def __init__(self):
        pass


class CGM1_GPS(ScoreProcedure):
    """Gait profile score accordin Baker et al. 2009

    Args:
        pointSuffix (str,Optional[None]): suffix added to model ouputs

    **Reference**

    Baker, Richard; McGinley, Jennifer L.; Schwartz, Michael H.; Beynon, Sarah; Rozumalski, Adam; Graham, H. Kerr; Tirosh, Oren (2009) The gait profile score and movement analysis profile. In : Gait & posture, vol. 30, n° 3, p. 265–269. DOI: 10.1016/j.gaitpost.2009.05.020.

    """


    def __init__(self,pointSuffix=None):
        super(CGM1_GPS, self).__init__()


        pointSuffix = ("_"+pointSuffix)  if (pointSuffix is not None) else ""

        matchingNormativeDataLabel = dict()

        matchingNormativeDataLabel["LPelvisAngles"+pointSuffix,"Left"] =  "PelvisAngles"
#        matchingNormativeDataLabel["RPelvisAngles"+pointSuffix,"Right"]=  "Pelvis.Angles"   # dont use. see richard`s articles
        matchingNormativeDataLabel["LHipAngles"+pointSuffix,"Left"]=  "HipAngles"
        matchingNormativeDataLabel["RHipAngles"+pointSuffix,"Right"]=  "HipAngles"
        matchingNormativeDataLabel["LKneeAngles"+pointSuffix,"Left"]=  "KneeAngles"
        matchingNormativeDataLabel["RKneeAngles"+pointSuffix,"Right"]=  "KneeAngles"
        matchingNormativeDataLabel["LAnkleAngles"+pointSuffix,"Left"]=  "AnkleAngles"
        matchingNormativeDataLabel["RAnkleAngles"+pointSuffix,"Right"]=  "AnkleAngles"
        matchingNormativeDataLabel["LFootProgressAngles"+pointSuffix,"Left"]=  "FootProgressAngles"
        matchingNormativeDataLabel["RFootProgressAngles"+pointSuffix,"Right"]=  "FootProgressAngles"

        axes={"PelvisAngles":[0,1,2],"HipAngles":[0,1,2],"KneeAngles":[0],"AnkleAngles":[0],"FootProgressAngles":[2]}   # tip is to use label from normative dataset


        self.matchingNormativeDataLabel = matchingNormativeDataLabel
        self.axes = axes

    def _compute(self,analysis,normativeData):

        gvs = dict()

        nLeftCycles,nRightCycles = analysis.getKinematicCycleNumbers()

        # --- MAP ---
        # left cycles
        for label,context in self.matchingNormativeDataLabel.keys():
            matchingNormativeDataLabel = self.matchingNormativeDataLabel[label,context]
            if context == "Left":
                left_rms_local = np.zeros((nLeftCycles,3))
                for i in range(0,nLeftCycles):
                    values = analysis.kinematicStats.data[label, context]["values"][i]
                    valuesNorm = normativeData[matchingNormativeDataLabel]["mean"]

                    if valuesNorm.shape[0] == 51:
                        rms = numeric.rms(values[0:101:2]-valuesNorm,axis=0)
                    else:
                        rms = numeric.rms(values-valuesNorm,axis=0)

                    left_rms_local[i,:] = rms

                gvs[label,context]=left_rms_local


        # right cycles
        for label,context in self.matchingNormativeDataLabel.keys():
            matchingNormativeDataLabel = self.matchingNormativeDataLabel[label,context]
            if context == "Right":
                right_rms_local = np.zeros((nRightCycles,3))
                for i in range(0,nRightCycles):
                    values = analysis.kinematicStats.data[label, context]["values"][i]
                    valuesNorm = normativeData[matchingNormativeDataLabel]["mean"]

                    if valuesNorm.shape[0] == 51:
                        rms = numeric.rms(values[0:101:2]-valuesNorm,axis=0)
                    else:
                        rms = numeric.rms(values-valuesNorm,axis=0)

                    right_rms_local[i,:] = rms

                gvs[label,context]=right_rms_local



        # --- GPS ---

        # number of axis. 15 according articles ( left )
        n_axis =0
        for axis in self.axes:
            n_axis = n_axis + len(self.axes[axis])

        # computation of rms for left cycles
        left_rms_global = np.zeros((nLeftCycles,n_axis))
        for i in range(0,nLeftCycles):
            cumAxis =0
            for label,context in  gvs.keys():
                if context == "Left":
                    axisIndex = self.axes[self.matchingNormativeDataLabel[label,context]] # use of the tip here
                    for axis in axisIndex:
                        left_rms_global[i,cumAxis] = gvs[label,context][i,axis]
                        cumAxis+=1

        # computation of rms for right cycles
        right_rms_global = np.zeros((nRightCycles,n_axis))
        for i in range(0,nRightCycles):
            cumAxis =0
            for label,context in  gvs.keys():
                if context == "Right":
                    axisIndex = self.axes[self.matchingNormativeDataLabel[label,context]]
                    for axis in axisIndex:
                        right_rms_global[i,cumAxis] = gvs[label,context][i,axis]
                        cumAxis+=1


        # output dictionary
        outDict_gvs = dict()
        outDict_gps_context = dict()
        outDict_gps = dict()

        for label,context in self.matchingNormativeDataLabel.keys():
            outDict_gvs[label,context]={'mean':np.mean(gvs[label, context],axis=0),
                                          'std':np.std(gvs[label, context],axis=0),
                                          'median': np.median(gvs[label, context],axis=0),
                                          'values':gvs[label,context]}
        for context in ["Left","Right"]:
            gpsValues = left_rms_global if context == "Left" else right_rms_global
            # construction of the global dictionary outputs
            outDict_gps_context[context]={'mean':np.array([np.mean(gpsValues.mean(axis=1))]),
                                      'std':np.array([np.std(gpsValues.mean(axis=1))]),
                                       'median': np.array([np.median(gpsValues.mean(axis=1))]),
                                       'values': gpsValues.mean(axis=1)}


        overall_gps_values = np.concatenate((outDict_gps_context["Right"]["values"],outDict_gps_context["Left"]["values"]))


        outDict_gps={'mean':np.array([np.mean(overall_gps_values)]),
                          'std':np.array([np.std(overall_gps_values)]),
                           'median': np.array([np.median(overall_gps_values)]),
                           'values': overall_gps_values}

        return outDict_gvs, outDict_gps_context,outDict_gps

## TODO:
#class GDI(ScoreProcedure):
#    def __init__(self):
#        super(GDI, self).__init__()
#        self.name = "Gait Deviation index"

#    def _compute(self):
#        pass
