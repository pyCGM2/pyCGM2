# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

from pyCGM2.Processing import exporter
from pyCGM2.Signal.detect_peaks import detect_peaks
from pyCGM2.Math import derivation


# --- abstract procedure
class DiscretePointProcedure(object):
    def __init__(self):
        pass

# --- PROCEDURE ----
class BenedettiProcedure(DiscretePointProcedure):
    """ discrete points recommanded by benededdi et al(1998).

    Args:
        pointSuffix (str): suffix added to model ouputs


    **References**:

    Benedetti, M. G.; Catani, F.; Leardini, A.; Pignotti, E.; Giannini, S. (1998) Data management in gait analysis for clinical applications. In : Clinical biomechanics (Bristol, Avon), vol. 13, n° 3, p. 204–215. DOI: 10.1016/s0268-0033(97)00041-7.


    """




    NAME = "Benedetti"


    def __init__(self,pointSuffix=None):
        super(BenedettiProcedure, self).__init__()

        self.pointSuffix = str("_"+pointSuffix)  if pointSuffix is not None else ""

    def detect (self,analysisInstance):
        """extract discrete points

        Args:
            analysisInstance (pyCGM2.Processing.analysis.Analysis): an `analysis` instance

        """

        # self.__detectTest(analysisInstance,"RHipMoment","Right") # TEST

        dataframes = list()
        # Left
        dataframes.append( self.__getPelvis_kinematics(analysisInstance,"LPelvisAngles","Left"))
        dataframes.append( self.__getHip_kinematics(analysisInstance,"LHipAngles","Left"))
        dataframes.append( self.__getKnee_kinematics(analysisInstance,"LKneeAngles","Left"))
        dataframes.append( self.__getAnkle_kinematics(analysisInstance,"LAnkleAngles","Left"))

        try:
            dataframes.append( self.__getHip_kinetics(analysisInstance,"LHipMoment","Left"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getKnee_kinetics(analysisInstance,"LKneeMoment","Left"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getAnkle_kinetics(analysisInstance,"LAnkleMoment","Left"))
        except KeyError:
            pass


        # Right
        dataframes.append( self.__getPelvis_kinematics(analysisInstance,"RPelvisAngles","Right"))
        dataframes.append( self.__getHip_kinematics(analysisInstance,"RHipAngles","Right"))
        dataframes.append( self.__getKnee_kinematics(analysisInstance,"RKneeAngles","Right"))
        dataframes.append( self.__getAnkle_kinematics(analysisInstance,"RAnkleAngles","Right"))


        try:
            dataframes.append( self.__getHip_kinetics(analysisInstance,"RHipMoment","Right"))
        except KeyError:
            pass


        try:
            dataframes.append( self.__getKnee_kinetics(analysisInstance,"RKneeMoment","Right"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getAnkle_kinetics(analysisInstance,"RAnkleMoment","Right"))
        except KeyError:
            pass



        return pd.concat(dataframes)

    def __construcPandasSerie(self,pointLabel,context, axis, cycleIndex,
                              discretePointProcedure,discretePointLabel,discretePointValue,discretePointDescription,
                              comment):
        iDict = OrderedDict([('VariableLabel', pointLabel),
                     ('EventContext', context),
                     ('Axis', axis),
                     ('Cycle', cycleIndex),
                     ('DiscretePointProcedure', discretePointProcedure),
                     ('Label', pointLabel[0]+discretePointLabel),
                     ('Value', discretePointValue),
                     ('DiscretePointDescription', discretePointDescription),
                     ('Comment', comment)])
        return pd.Series(iDict)


    def __detectTest(self,analysisInstance,pointLabel,context):

        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        # experiment with detect_peaks, marcos's function
        i=1

        fullValues = normalizedCycleValues["values"][i][:,1]

        begin = 0
        end  = 101
        values = normalizedCycleValues["values"][i][begin:end,1]
        indexes = detect_peaks(values,show = False, valley=False)
        plt.figure()
        plt.plot(fullValues)
        plt.axvline(int(stanceValues[i]),color='r',ls='dashed')
        for it in indexes:
            plt.plot(begin+it,fullValues[begin+it],"+r",mew=2, ms=8)


    def __getPelvis_kinematics(self,analysisInstance,pointLabel,context):

        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()

        #---min rotation sagital plane
        label = ["HR1","THR1"]
        axis = "X"
        desc = ["min rotation sagital plane","frame of min rotation sagital plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][:,0])
            frame = np.argmin(normalizedCycleValues["values"][i][:,0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---min rot coronal plane ( erreur dans la table 1 de benedetti)
        label = ["HR2","THR2"]
        axis = "Y"
        desc = ["min rot coronal plane","frame of min rot coronal plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][:,1])
            frame = np.argmin(normalizedCycleValues["values"][i][:,1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---max rot coronal plane
        label = ["HR3","THR3"]
        axis = "Y"
        desc = ["max rot coronal plane","frame of max rot coronal plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1])
            frame = np.argmax(normalizedCycleValues["values"][i][:,1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #--- max rot transverse plane
        label = ["HR4","THR4"]
        axis = "Y"
        desc = ["max rot transverse plane","frame of max rot transverse plane"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1])
            frame = np.argmax(normalizedCycleValues["values"][i][:,1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        return pd.DataFrame(series)

    def __getHip_kinematics(self,analysisInstance,pointLabel,context):


        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()

        #---flexion at heel strike
        label = "H1"
        axis = "X"
        desc = "flexion at heel strike"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][0,0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---flexion at loading response(H2)
        label = "H2"
        axis = "X"
        desc = "flexion at loading response"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(loadingResponseValues[i]),0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---extension max in stance(H3-TH3)
        label = ["H3","TH3"]
        axis = "X"
        desc = ["extension max in stance","frame of extension max in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        #---flexion at toe-off (H4)
        label = "H4"
        axis = "X"
        desc = "flexion at toe-off"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(stanceValues[i]),0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)




        #---max flexion in swing(H5-TH5)
        label = ["H5","TH5"]
        axis = "X"
        desc = ["max flexion in swing","frame of max flexion in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)




        #---total sagital plane excursion(H6))
        label = "H6"
        axis = "X"
        desc = "total sagital plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,0]) - np.min(normalizedCycleValues["values"][i][:,0])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---total coronal plane excursion(H7))
        label = "H7"
        axis = "Y"
        desc = "total coronal plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1]) - np.min(normalizedCycleValues["values"][i][:,1])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---max adduction in stance(H8-TH8))
        label = ["H8","TH8"]
        axis = "Y"
        desc = ["max adduction in stance","frame of max adduction in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)





        #---max abd in swing(H9-TH9))
        label = ["H9","TH9"]
        axis = "Y"
        desc = ["max abduction in swing","frame of max abduction in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1])
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



        #---total transverse plane excursion(H10))
        label = "H10"
        axis = "Z"
        desc = "total transverse plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,2]) - np.min(normalizedCycleValues["values"][i][:,2])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---max rot int in stance(H10-TH10))
        label = ["H11","TH11"]
        axis = "Z"
        desc = ["max rot int in stance","frame of max rot int in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---max rot ext in swing(H11-TH11))
        label = ["H12","TH12"]
        axis = "Z"
        desc = ["max rot ext in swing","frame of max rot ext in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2])
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



        return pd.DataFrame(series)


    def __getKnee_kinematics(self,analysisInstance,pointLabel,context):


        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()


        #---flexion at heel strike
        label = "K1"
        axis = "X"
        desc = "flexion at heel strike"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][0,0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---flexion at loading response
        label = "K2"
        axis = "X"
        desc = "flexion at loading response"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(loadingResponseValues[i]),0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---extension max in stance
        label = ["K3","TK3"]
        axis = "X"
        desc = ["extension max in stance","frame of extension max in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---flexion at toe-off
        label = "K4"
        axis = "X"
        desc = "flexion at toe-off"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(stanceValues[i]),0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---max flexion in swing
        label = ["K5","TK5"]
        axis = "X"
        desc = ["max flexion in swing","frame of max flexion in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)




        #---total sagital plane excursion
        label = "K6"
        axis = "X"
        desc = "total sagital plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,0]) - np.min(normalizedCycleValues["values"][i][:,0])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---total coronal plane excursion
        label = "K7"
        axis = "Y"
        desc = "total coronal plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,1]) - np.min(normalizedCycleValues["values"][i][:,1])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---max adduction in stance(H8-TH8))
        label = ["K8","TK8"]
        axis = "Y"
        desc = ["max adduction in stance","frame of max adduction in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)





        #---max abd in swing(H9-TH9))
        label = ["K9","TK9"]
        axis = "Y"
        desc = ["max abduction in swing","frame of max abduction in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1])
            frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



        #---total transverse plane excursion
        label = "K10"
        axis = "Z"
        desc = "total transverse plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,2]) - np.min(normalizedCycleValues["values"][i][:,2])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #--max rot int in stance
        label = ["K11","TK11"]
        axis = "Z"
        desc = ["max rot int in stance","frame of max rot int in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)

        #---max rot ext in swing
        label = ["K12","TK12"]
        axis = "Z"
        desc = ["max rot ext in swing","frame of max rot ext in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2])
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,2])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)



        return pd.DataFrame(series)

    def __getAnkle_kinematics(self,analysisInstance,pointLabel,context):


        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kinematicStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()


        #---flexion at heel strike
        label = "A1"
        axis = "X"
        desc = "flexion at heel strike"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][0,0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #---flexion at loading response
        label = "A2"
        axis = "X"
        desc = "flexion at loading response"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(loadingResponseValues[i]),0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #---dorsi flex max in stance
        label = ["A3","TA3"]
        axis = "X"
        desc = ["dorsi flex max in stance","frame of dorsi flex max in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        #---flexion at toe-off
        label = "A4"
        axis = "X"
        desc = "flexion at toe-off"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = normalizedCycleValues["values"][i][int(stanceValues[i]),0]

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)


        #--- 5- max plant flexion in swing ( error in benedetti table) !!!

        #  rule :
        #    - find the first peak inferior to value at TO from pre-swing(ie TO reduced by 10 frames)
        # note :
        #    - too dependant to detection of the TO frame. Maximal plantar flexion can occurs before Toe off !
        #   - use detect_peak

        label = ["A5","TA5"]
        axis = "X"
        desc = ["max plant flexion in swing","frame of max plant flexion in swing"]

        for i in range(0,len(normalizedCycleValues["values"])):

            threshold = 10
            frameTO = int(stanceValues[i])
            beginFrame = frameTO-threshold
            toeOffValue = normalizedCycleValues["values"][i][frameTO,0]

            values = normalizedCycleValues["values"][i][:,0]
            valuesFromPreswing = normalizedCycleValues["values"][i][beginFrame:101,0]
            indexes = detect_peaks(valuesFromPreswing, valley=True)

            # rule application
            frame = "NA"
            value = "NA"
            comment =""
            if indexes != []:
                for ind in indexes:
                    if values[ind]<toeOffValue:
                        frame = beginFrame + int(ind)
                        value = values[frame]
                        if frame < frameTO: comment = " warning : value before TO"
                        break
                    else:
                        frame = "NA"
                        value = "NA"
                        comment = "no peak found"


            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               comment)
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               comment)
            series.append(serie)

        #---6 total sagital plane excursion !!
        # search for max/min without regard to phases
        # note : min might be different from A5 if min occurs before preswing (TO-10 frames)

        label = "A6"
        axis = "X"
        desc = "total sagital plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][:,0]) - np.min(normalizedCycleValues["values"][i][:,0])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #--- 7 total coronal plane excursion !!!
        # note :
        #   -max in stance (ie. A7) minus min in swing (ie. A8)
        label = "A7"
        axis = "Y"
        desc = "total coronal plane excursion"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1]) - np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1])
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label,value,desc,
                                               "")
            series.append(serie)

        #--- 8 max inversion in stance
        label = ["A8","TA8"]
        axis = "Y"
        desc = ["max inversion in stance","frame of max inversion in stance"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])
            frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        #-- 9 max eversion in swing
        label = ["A9","TK9"]
        axis = "Y"
        desc = ["max eversion in swing","frame of max eversion in swing"]
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1])
            frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,1])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],value,desc[0],
                                               "")
            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frame,desc[1],
                                               "")
            series.append(serie)


        return pd.DataFrame(series)


    def __getHip_kinetics(self,analysisInstance,pointLabel,context):


        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()


        #---1-max and 2-min extensor moments

        for i in range(0,len(normalizedCycleValues["values"])):

            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            valueMax = np.max(normalizedCycleValues["values"][i][0:frameMin+1,0])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:frameMin+1,0])

            label = ["HM1","THM1"]
            axis = "X"
            desc = ["max flex moment", "frame of max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[1],
                                               "")

            series.append(serie)

            label = ["HM2","THM2"]
            axis = "X"
            desc = ["max ext moment", "frame of max ext moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[1],
                                               "")

            series.append(serie)


        #---3 first and 4 second abductor moments !!
        # rule :
        #  - find  first max during the first mid stance
        # - find the second max from the first max frame
        # note :
        #  - apparently Benedetti aductor curve is not conventional, compared wth plot from gait books

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax1 = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])
            frameMax1 = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])

            valueMax2 = np.max(normalizedCycleValues["values"][i][frameMax1:int(stanceValues[i]),1])
            frameMax2 = frameMax1 + np.argmax(normalizedCycleValues["values"][i][frameMax1:int(stanceValues[i]),1])

            label = ["HM3","THM3"]
            axis = "Y"
            desc = ["first max abductor moment", "frame of the first max abductor moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax1,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax1,desc[1],
                                               "")

            series.append(serie)

            label = ["HM4","THM4"]
            axis = "Y"
            desc = ["second max abductor moment", "frame of the second max abductor moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax2,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax2,desc[1],
                                               "")

            series.append(serie)


        #---5 -min and 6-max rotation moment
        for i in range(0,len(normalizedCycleValues["values"])):

            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])

            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])

            label = ["HM5","THM5"]
            axis = "Z"
            desc = ["max ext rot moment", "frame of max ext rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[1],
                                               "")

            series.append(serie)

            label = ["HM6","THM6"]
            axis = "Z"
            desc = ["max int rot moment", "frame of max int rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[1],
                                               "")

            series.append(serie)

        return pd.DataFrame(series)


    def __getKnee_kinetics(self,analysisInstance,pointLabel,context):


        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()

        #---1 fist flexor extensor, 2- max extensor, 3-second max flexor moment !!!
        # rule :
        #  - detect extensor max firstly and find first and second max flexor subsequently
        # note :
        #  - benededdi confuse flex and extensor moment apparently

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            if frameMax !=0:
                valueFistMin = np.min(normalizedCycleValues["values"][i][0:int(frameMax),0])
                frameFistMin = np.argmin(normalizedCycleValues["values"][i][0:int(frameMax),0])
                commentFistMin =""
            else:
                valueFistMin ="NA"
                frameFistMin = "NA"
                commentFistMin = "Warning first extensor dtected at frame 0"



            valueSecondMin = np.min(normalizedCycleValues["values"][i][int(frameMax):int(stanceValues[i]),0])
            frameSecondMin = int(frameMax) + np.argmin(normalizedCycleValues["values"][i][int(frameMax):int(stanceValues[i]),0])


            label = ["KM1","TKM1"]
            axis = "X"
            desc = ["first max flex moment", "frame of first max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueFistMin,desc[0],
                                               commentFistMin)

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameFistMin,desc[0],
                                               commentFistMin)

            series.append(serie)

            label = ["KM2","TKM2"]
            axis = "X"
            desc = ["max extensor moment", "frame of max ext moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[0],
                                               "")

            series.append(serie)


            label = ["KM3","TKM3"]
            axis = "X"
            desc = ["second max flex moment", "frame of second max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueSecondMin,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameSecondMin,desc[0],
                                               "")

            series.append(serie)

        #---4 max adductor, 5- first max abd , 6- second max abd  moment
        # rule :
        #   - find first max abd on mid stance
        #   - find max add from 0 to first max abd
        #   - find second abd max from max abd to TO

        # note :
        #  - Benedetti illustred an opposite trace

        for i in range(0,len(normalizedCycleValues["values"])):

            valueFirstMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])
            frameFirstMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]/2.0),1])


            if frameFirstMax !=0:
                valueMin = np.min(normalizedCycleValues["values"][i][0:int(frameFirstMax),1])
                frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(frameFirstMax),1])
                commentMin =""
            else:
                valueMin,frameMin ="NA","NA"
                commentMin = "warning first max detected at 0"


            valueSecondMax = np.max(normalizedCycleValues["values"][i][int(frameFirstMax):int(stanceValues[i]),1])
            frameSecondMax = int(frameFirstMax) + np.argmax(normalizedCycleValues["values"][i][int(frameFirstMax):int(stanceValues[i]),1])


            label = ["KM4","TKM4"]
            axis = "Y"
            desc = ["max add moment", "frame of max add moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               commentMin)

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[0],
                                               commentMin)

            series.append(serie)


            label = ["KM5","TKM5"]
            axis = "Y"
            desc = ["first max abd moment", "frame of first max abd moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueFirstMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameFirstMax,desc[0],
                                               "")

            series.append(serie)

            label = ["KM6","TKM6"]
            axis = "Y"
            desc = ["second max abd moment", "frame of second max abd moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueSecondMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameSecondMax,desc[0],
                                               "")

            series.append(serie)

        #-- 7 max int, 8- max ext rotation moment
        for i in range(0,len(normalizedCycleValues["values"])):
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])

            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])


            label = ["KM7","TKM7"]
            axis = "Z"
            desc = ["max int rot moment", "frame of max int rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[0],
                                               "")

            series.append(serie)

            label = ["KM8","TKM8"]
            axis = "Z"
            desc = ["max ext rot moment", "frame of max ext rot moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[0],
                                               "")

            series.append(serie)

        return pd.DataFrame(series)


    def __getAnkle_kinetics(self,analysisInstance,pointLabel,context):


        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]
        loadingResponseValues = analysisInstance.kineticStats.pst['doubleStance1', context]['values']
        stanceValues =         analysisInstance.kineticStats.pst['stancePhase', context]['values']

        series = list()

        #---1 max flex , 2 max ext moment

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])


            label = ["AM1","TAM1"]
            axis = "X"
            desc = [" max flex moment", "frame of max flex moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[0],
                                               "")

            label = ["AM2","TAM2"]
            axis = "X"
            desc = [" max ext moment", "frame of max ext moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[0],
                                               "")

        #--4 max eversor, 5 max inv moment

        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMax = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])

            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])
            frameMin = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i]),2])


            label = ["AM3","TAM3"]
            axis = "Y"
            desc = [" max ever moment", "frame of max ever moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMax,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMax,desc[0],
                                               "")

            label = ["AM4","TAM4"]
            axis = "Y"
            desc = [" max inv moment", "frame of max inv moment"]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[0],valueMin,desc[0],
                                               "")

            series.append(serie)

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                               int(i),
                                               BenedettiProcedure.NAME,
                                               label[1],frameMin,desc[0],
                                               "")

            series.append(serie)

        return pd.DataFrame(series)


class MaxMinProcedure(DiscretePointProcedure):
    """ extract extrema values.

    Args:
        pointSuffix (str): suffix added to model ouputs

    """
    NAME = "MaxMin"


    def __init__(self,pointSuffix=None):
        super(MaxMinProcedure, self).__init__()
        self.pointSuffix = str("_"+pointSuffix)  if pointSuffix is not None else ""

    def detect (self,analysisInstance):
        """extract discrete points

        Args:
            analysisInstance (pyCGM2.Processing.analysis.Analysis): an `analysis` instance

        """

        dataframes = list()
        # Left
        dataframes.append( self.__getExtrema(analysisInstance,"LPelvisAngles","Left"))

        dataframes.append( self.__getExtrema(analysisInstance,"LPelvisAngles","Left"))
        dataframes.append( self.__getExtrema(analysisInstance,"LHipAngles","Left"))
        dataframes.append( self.__getExtrema(analysisInstance,"LKneeAngles","Left"))
        dataframes.append( self.__getExtrema(analysisInstance,"LAnkleAngles","Left"))

        try:
            dataframes.append( self.__getExtrema(analysisInstance,"LHipMoment","Left", dataType = "Kinetics"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getExtrema(analysisInstance,"LKneeMoment","Left", dataType = "Kinetics"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getExtrema(analysisInstance,"LAnkleMoment","Left", dataType = "Kinetics"))
        except KeyError:
            pass



        # Right
        dataframes.append( self.__getExtrema(analysisInstance,"RPelvisAngles","Right"))

        dataframes.append( self.__getExtrema(analysisInstance,"RHipAngles","Right"))
        dataframes.append( self.__getExtrema(analysisInstance,"RKneeAngles","Right"))
        dataframes.append( self.__getExtrema(analysisInstance,"RAnkleAngles","Right"))

        try:
            dataframes.append( self.__getExtrema(analysisInstance,"RHipMoment","Right", dataType = "Kinetics"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getExtrema(analysisInstance,"RKneeMoment","Right", dataType = "Kinetics"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getExtrema(analysisInstance,"RAnkleMoment","Right", dataType = "Kinetics"))
        except KeyError:
            pass
        return pd.concat(dataframes)


        return pd.concat(dataframes)

    def __construcPandasSerie(self,pointLabel,context, axis, cycleIndex,
                              discretePointProcedure,discretePointLabel,discretePointValue,discretePointDescription,
                              comment):
        iDict = OrderedDict([('VariableLabel', pointLabel),
                     ('EventContext', context),
                     ('Axis', axis),
                     ('Cycle', cycleIndex),
                     ('DiscretePointProcedure', discretePointProcedure),
                     ('Label', pointLabel[0]+discretePointLabel),
                     ('Value', discretePointValue),
                     ('DiscretePointDescription', discretePointDescription),
                     ('Comment', comment)])
        return pd.Series(iDict)


    def __getExtrema(self,analysisInstance,pointLabel,context,dataType="Kinematics"):

        if dataType == "Kinematics":
            normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel+self.pointSuffix,context]
        if dataType == "Kinetics":
            normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel+self.pointSuffix,context]

        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()

        #---min stance
        label = ["minST","TminST"]
        axes = ["X","Y","Z"]

        for axInd in range(0,len(axes)):

            desc = ["min stance","frame of min stance"]
            for i in range(0,len(normalizedCycleValues["values"])):
                value = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i])+1,axInd])
                frame = np.argmin(normalizedCycleValues["values"][i][0:int(stanceValues[i])+1,axInd])

                serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                   int(i),
                                                   MaxMinProcedure.NAME,
                                                   label[0],value,desc[0],
                                                   "")
                series.append(serie)

                serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                   int(i),
                                                   MaxMinProcedure.NAME,
                                                   label[1],frame,desc[1],
                                                   "")
                series.append(serie)

                #---max stance
                label = ["maxST","TmaxST"]
                desc = ["max stance","frame of max stance"]
                for i in range(0,len(normalizedCycleValues["values"])):
                    value = np.max(normalizedCycleValues["values"][i][0:int(stanceValues[i])+1,axInd])
                    frame = np.argmax(normalizedCycleValues["values"][i][0:int(stanceValues[i])+1,axInd])

                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       MaxMinProcedure.NAME,
                                                       label[0],value,desc[0],
                                                       "")
                    series.append(serie)

                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       MaxMinProcedure.NAME,
                                                       label[1],frame,desc[1],
                                                       "")
                    series.append(serie)

                 #---min swing
                label = ["minSW","TminSW"]
                desc = ["min swing","frame of min swing"]
                for i in range(0,len(normalizedCycleValues["values"])):
                    value = np.min(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])
                    frame = int(stanceValues[i]) + np.argmin(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])

                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[0],value,desc[0],
                                                       "")
                    series.append(serie)

                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[1],frame,desc[1],
                                                       "")
                    series.append(serie)

                #---max swing
                label = ["maxSW","TmaxSW"]
                desc = ["max swing","frame of max swing"]
                for i in range(0,len(normalizedCycleValues["values"])):
                    value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])
                    frame = int(stanceValues[i]) + np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,axInd])

                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[0],value,desc[0],
                                                       "")
                    series.append(serie)

                    serie = self.__construcPandasSerie(pointLabel,context,axes[axInd],
                                                       int(i),
                                                       BenedettiProcedure.NAME,
                                                       label[1],frame,desc[1],
                                                       "")
                    series.append(serie)

        return pd.DataFrame(series)


class GoldbergProcedure(DiscretePointProcedure):
    """ discrete points recommanded by Goldberg et al(1998).

    Args:
        pointSuffix (str): suffix added to model ouputs

    **References**:

    Goldberg, Saryn R.; Ounpuu, Sylvia; Arnold, Allison S.; Gage, James R.; Delp, Scott L. (2006) Kinematic and kinetic factors that correlate with improved knee flexion following treatment for stiff-knee gait. In : Journal of biomechanics, vol. 39, n° 4, p. 689–698. DOI: 10.1016/j.jbiomech.2005.01.015.

    """

    NAME = "Goldberg"


    def __init__(self,pointSuffix=None):
        super(GoldbergProcedure, self).__init__()
        self.pointSuffix = str("_"+pointSuffix)  if pointSuffix is not None else ""

    def detect (self,analysisInstance):
        """extract discrete points

        Args:
            analysisInstance (pyCGM2.Processing.analysis.Analysis): an `analysis` instance

        """

        dataframes = list()

        dataframes.append( self.__getKnee_kinematics(analysisInstance,"LKneeAngles"+self.pointSuffix,"Left"))
        dataframes.append( self.__getKnee_kinematics(analysisInstance,"RKneeAngles"+self.pointSuffix,"Right"))

        try:
            dataframes.append( self.__getKnee_kinetics(analysisInstance,"LKneeMoment"+self.pointSuffix,"LKneeAngles"+self.pointSuffix,"Left"))
        except KeyError:
            pass

        try:
            dataframes.append( self.__getKnee_kinetics(analysisInstance,"RKneeMoment"+self.pointSuffix,"RKneeAngles"+self.pointSuffix,"Right"))
        except KeyError:
            pass


        return pd.concat(dataframes)

    def __construcPandasSerie(self,pointLabel,context, axis, cycleIndex,
                              discretePointProcedure,discretePointLabel,discretePointValue,discretePointDescription,
                              comment):
        iDict = OrderedDict([('VariableLabel', pointLabel),
                     ('EventContext', context),
                     ('Axis', axis),
                     ('Cycle', cycleIndex),
                     ('DiscretePointProcedure', discretePointProcedure),
                     ('Label', pointLabel[0]+discretePointLabel),
                     ('Value', discretePointValue),
                     ('DiscretePointDescription', discretePointDescription),
                     ('Comment', comment)])
        return pd.Series(iDict)


    def __getKnee_kinematics(self,analysisInstance,pointLabel,context):

        normalizedCycleValues = analysisInstance.kinematicStats.data [pointLabel,context]
        stanceValues =         analysisInstance.kinematicStats.pst['stancePhase', context]['values']

        series = list()

        #---maximal knee flexion
        label = "G1"
        axis = "X"

        desc = "max knee flexion in swing"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])


            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                                   int(i),
                                                   GoldbergProcedure.NAME,
                                                   label,value,desc,
                                                   "")
            series.append(serie)


        #---range of knee flexion in early swing
        label = "G2"
        axis = "X"

        desc = "range knee flexion in  early swing"
        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            valueTO = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]),0])

            value = valueMax-valueTO

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                                   int(i),
                                                   GoldbergProcedure.NAME,
                                                   label,value,desc,
                                                   "")
            series.append(serie)

        #---total range of knee motion
        label = "G3"
        axis = "X"

        desc = "total range knee motion"
        for i in range(0,len(normalizedCycleValues["values"])):
            valueMax = np.max(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])
            valueMin = np.min(normalizedCycleValues["values"][i][0:int(stanceValues[i]),0])

            value = valueMax-valueMin

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                                   int(i),
                                                   GoldbergProcedure.NAME,
                                                   label,value,desc,
                                                   "")
            series.append(serie)

        #---timing of peak knee flexion relative to TO
        label = "G4"
        axis = "X"

        desc = "timing of peak knee flexion"
        for i in range(0,len(normalizedCycleValues["values"])):

            frame = np.argmax(normalizedCycleValues["values"][i][int(stanceValues[i]):101,0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                                   int(i),
                                                   GoldbergProcedure.NAME,
                                                   label,frame,desc,
                                                   "")
            series.append(serie)

        #---velocity at TO
        label = "G5"
        axis = "X"

        desc = "velocity at TO"
        for i in range(0,len(normalizedCycleValues["values"])):



            derivativeValues = derivation.firstOrderFiniteDifference(normalizedCycleValues["values"][i],1.0)

            value = derivativeValues[int(stanceValues[i]),0]
            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                                   int(i),
                                                   GoldbergProcedure.NAME,
                                                   label,value,desc,
                                                   "")
            series.append(serie)


        return pd.DataFrame(series)


    def __getKnee_kinetics(self,analysisInstance,pointLabel,kinematicPointLabel,context):

        normalizedCycleValues = analysisInstance.kineticStats.data [pointLabel,context]
        normalizedKinematicCycleValues =     analysisInstance.kineticStats.optionalData[kinematicPointLabel,context]

        stanceFrame =         analysisInstance.kineticStats.pst['stancePhase', context]['values']
        secondDoubleStanceFrameRange =         analysisInstance.kineticStats.pst["doubleStance2",context]['values']
        doubleStanceFrame = stanceFrame-secondDoubleStanceFrameRange


        series = list()

        #---average moment in double stance
        label = "G6"
        axis = "X"

        desc = "knee moment average in double stance"
        for i in range(0,len(normalizedCycleValues["values"])):
            value = np.mean(normalizedCycleValues["values"][i][int(doubleStanceFrame[i]):int(stanceFrame[i])+1,0])


            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                                   int(i),
                                                   GoldbergProcedure.NAME,
                                                   label,value,desc,
                                                   "")
            series.append(serie)

        #---average moment in early swing
        label = "G7"
        axis = "X"

        desc = "knee moment average in early swing"
        for i in range(0,len(normalizedCycleValues["values"])):

            earlySwingFrame = int(stanceFrame[i]) + np.argmax(normalizedKinematicCycleValues["values"][i][int(stanceFrame[i]):101,0])
            value = np.mean(normalizedCycleValues["values"][i][int(stanceFrame[i]):int(earlySwingFrame)+1,0])

            serie = self.__construcPandasSerie(pointLabel,context,axis,
                                                   int(i),
                                                   GoldbergProcedure.NAME,
                                                   label,value,desc,
                                                   "")
            series.append(serie)

        return pd.DataFrame(series)

# class XlsProcedure(object):
#
#
#
#     def __init__(self,xlsFiles,ruleEnable=False,pointSuffix=None):
#
#         self.pointSuffix = str("_"+pointSuffix)  if pointSuffix is not None else ""
#         self.m_rules = pd.read_excel(xlsFiles)
#         self.m_enableRules = ruleEnable
#
#     def detect(self,analysisInstance):
#         if self.m_enableRules:
#             df = self._extractWithRules(analysisInstance)
#         else:
#             df = self._extract(analysisInstance)
#         return df
#
#
#     def __construcPandasSerie(self,id,domain,cyclePeriod, variable, side,plan,detail,value):
#         iDict = OrderedDict([
#                      ('Id', id),
#                      ('Domain', domain),
#                      ('CyclePeriod', cyclePeriod),
#                      ('Variable', variable),
#                      ('Side', side),
#                      ('Plan', plan),
#                      ('Details', detail),
#                      ('Value', value)])
#         return pd.Series(iDict)
#
#     def __construcPandasSerie_withRules(self,id,domain, name, side,rule,comment,value):
#         iDict = OrderedDict([
#                      ('Id', id),
#                      ('Domain', domain),
#                      ('Name', name),
#                      ('Side', side),
#                      ('Rules', rule),
#                      ('Comment', comment),
#                      ('Value', value)])
#         return pd.Series(iDict)
#
#     def _computeMethod(self, values, context, method, CyclePeriod, phases,methodArg):
#
#         if CyclePeriod == "GC":
#             frameLimits= [0, 100]
#         elif CyclePeriod == "St":
#             frameLimits= phases[context,"St"]
#         elif CyclePeriod == "Sw":
#             frameLimits= phases[context,"Sw"]
#         elif CyclePeriod == "DS1":
#             frameLimits= phases[context,"DS1"]
#         elif CyclePeriod == "SS":
#             frameLimits= phases[context,"SS"]
#         elif CyclePeriod == "DS2":
#             frameLimits= phases[context,"DS2"]
#         elif CyclePeriod == "ISw":
#             frameLimits= phases[context,"ISw"]
#         elif CyclePeriod == "MSw":
#             frameLimits= phases[context,"MSw"]
#         elif CyclePeriod == "TSw":
#             frameLimits= phases[context,"TSw"]
#         else:
#             raise Exception ("Cycle period doesn t recognize")
#
#
#         if method == "mean":
#             val = np.mean(values[frameLimits[0]:frameLimits[1]])
#         elif method == "range":
#             val = np.abs( np.max(values[frameLimits[0]:frameLimits[1]]) -
#                           np.min(values[frameLimits[0]:frameLimits[1]]))
#         elif method == "min":
#             val = np.min(values[frameLimits[0]:frameLimits[1]])
#         elif method == "max":
#             val = np.max(values[frameLimits[0]:frameLimits[1]])
#         elif method == "minDer":
#             val = np.min(np.diff(values[frameLimits[0]:frameLimits[1]]))
#         elif method == "maxDer":
#             val = np.max(np.diff(values[frameLimits[0]:frameLimits[1]]))
#
#         elif method == "peak":
#             args = methodArg.split(",")
#             vals= values[frameLimits[0]:frameLimits[1]]
#             indexes = detect_peaks(vals, mph=float(args[0]), mpd=float(args[1]),  show = False, valley=False)
#             if indexes.shape[0]  > 2:
#                 val = "True"
#             else:
#                 val = "False"
#
#         elif method == "speed":
#             val = np.mean(np.diff(values[frameLimits[0]:frameLimits[1]]))
#
#
#         else:
#             val = 0
#
#         return val
#
#     def _applyRules(self,value, row,ruleType,method):
#
#         pass_rules = False
#         comment = None
#
#         if method == "minDer":
#             if value <0:
#                 comment = row["Comment_Normal"].values[0]
#             else:
#                 comment = row["Comment_Low"].values[0]
#
#         if method == "maxDer":
#             if value >0:
#                 comment = row["Comment_Normal"].values[0]
#             else:
#                 comment = row["Comment_Low"].values[0]
#
#         if method == "peak":
#             if value == "False":
#                 comment = row["Comment_Normal"].values[0]
#             else:
#                 comment = row["Comment_Low"].values[0]
#
#         if ruleType == 1:
#             if value> row["Norm_Min"].values[0] and value<row["Norm_Max"].values[0]:
#                 comment = row["Comment_Normal"].values[0]
#                 pass_rules = True
#             elif value> row["Norm_Max"].values[0] and value<row["Norm_Low"].values[0]:
#                 comment = row["Comment_Low"].values[0]
#                 pass_rules = True
#             elif value> row["Norm_Low"].values[0] and value<row["Norm_Moderate"].values[0]:
#                 comment = row["Comment_Moderate"].values[0]
#                 pass_rules = True
#             elif value> row["Norm_Moderate"].values[0] :
#                 comment = row["Comment_High"].values[0]
#                 pass_rules = True
#
#         elif ruleType == 2:
#             if value> row["Norm_Max"].values[0] and value<row["Norm_Min"].values[0]:
#                 comment = row["Comment_Normal"].values[0]
#             elif value< row["Norm_Max"].values[0] and value>row["Norm_Low"].values[0]:
#                 comment = row["Comment_Low"].values[0]
#             elif value< row["Norm_Low"].values[0] and value>row["Norm_Moderate"].values[0]:
#                 comment = row["Comment_Moderate"].values[0]
#             elif value<row["Norm_Moderate"].values[0] :
#                 comment = row["Comment_High"].values[0]
#
#         return comment, pass_rules
#
#
#
#
#     def _extract(self,analysisInstance):
#         rules = self.m_rules
#
#         series = list()
#
#         kinematicData = analysisInstance.kinematicStats
#
#         #phases
#         kinematicPhases = analysisHandler.getPhases(kinematicData)
#
#         # dataframe iteration
#         indexes = rules.index
#
#         for index in indexes:
#             row = rules[rules["Id"] == index+1 ] # rules.iloc[i,:]
#
#             if row["Side"].values[0] == "L":
#                 contexts = ["Left"]
#             if row["Side"].values[0] == "R":
#                 contexts = ["Right"]
#             elif row["Side"].values[0] == "B":
#                 contexts = ["Left","Right"]
#
#             for context in contexts:
#                 side = "L" if context == "Left" else "R"
#                 variable = str(side +row["Variable"].values[0])
#                 domain = row["Domain"].values[0]
#                 plan = row["Plan"].values[0]
#                 detail = row["Details"].values[0]
#                 cyclePeriod = row["CyclePeriod"].values[0]
#                 method = row["Method"].values[0]
#                 methodArg = row["MethodArgument"].values[0]
#
#                 if domain == "kinematics":
#                     values  = kinematicData.data[variable,context]["mean"][:,plan]
#                     valueExtract = self._computeMethod(values,context,method,cyclePeriod,kinematicPhases,methodArg)
#
#                 serie = self.__construcPandasSerie(index,domain,cyclePeriod,variable, side,plan,detail,valueExtract)
#                 series.append(serie)
#
#
#         return  pd.concat(series,axis=1).transpose()
#
#     def _extractWithRules(self,analysisInstance):
#         rules = self.m_rules
#
#         series = list()
#
#         kinematicData = analysisInstance.kinematicStats
#
#         #phases
#         kinematicPhases = analysisHandler.getPhases(kinematicData)
#
#         # dataframe iteration
#         indexes = rules.index
#
#         for index in indexes:
#             row = rules[rules["Id"] == index+1 ] # rules.iloc[i,:]
#             if row["Activate"].values[0] == 1:
#                 if row["Side"].values[0] == "L":
#                     contexts = ["Left"]
#                 elif row["Side"].values[0] == "R":
#                     contexts = ["Right"]
#                 elif row["Side"].values[0] == "B":
#                     contexts = ["Left","Right"]
#
#                 for context in contexts:
#
#                     Id = row["Id"].values[0] # word id is a python keyword
#                     name = row["Name"].values[0] # word id is a python keyword
#
#                     # for getting values
#                     domain = row["Domain"].values[0]
#                     side = "L" if context == "Left" else "R"
#                     variable = str(side +row["Variable"].values[0])
#                     plan = row["Plan"].values[0]
#                     method = row["Method"].values[0]
#                     methodArg = row["MethodArgument"].values[0]
#                     cyclePeriod = row["CyclePeriod"].values[0]
#
#                     # for rule application
#                     rule = row["Rules"].values[0]
#                     ruleType = row["Type"].values[0]
#                     alternativeId = row["Alternative_rules"].values[0]
#
#                     if domain == "kinematics":
#                         values  = kinematicData.data[variable,context]["mean"][:,plan]
#                         valueExtract = self._computeMethod(values,context,method,cyclePeriod,kinematicPhases,methodArg)
#                         comment, pass_rules = self._applyRules(valueExtract,row,ruleType,method)
#
#                         if not pass_rules:
#                             if alternativeId>0:
#                                 rules.loc[rules["Id"] == alternativeId, "Activate"] = 1.0
#                                 LOGGER.logger.warning("rules (%i) activated" %(alternativeId) )
#
#                         if comment is not None:
#
#                             serie = self.__construcPandasSerie_withRules(Id,
#                             domain, name, side,rule,comment,valueExtract)
#                             series.append(serie)
#
#             else:
#                 print "No comment"
#
#         return  pd.concat(series,axis=1).transpose()
