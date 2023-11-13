
"""
Module contains procedure for extracting discrete value ( e.g : amplitude) from each emg signal
"""


import numpy as np
import pandas as pd

from collections import OrderedDict
from pyCGM2.Processing import exporter
from pyCGM2.Processing.analysis import Analysis


class DiscreteEmgProcedure(object):
    def __init__(self):
        pass

class AmplitudesProcedure(DiscreteEmgProcedure):
    """
    This procedure computes EMG amplitude for each gait phases
    """

    NAME = "EMG Amplitude from Integration for each gait phase"

    def __init__(self):
        super(AmplitudesProcedure, self).__init__()

    def detect(self, analysisInstance:Analysis, emgLabels:list[str], emgMuscles:list[str], emgContexts:list[str])->pd.DataFrame:
        """compute amplitudes

        Args:
            analysisInstance (Analysis): _description_
            emgLabels (list[str]): emg channels
            emgMuscles (list[str]): muscle names
            emgContexts (list[str]): event context

        Returns:
            pd.DataFrame: dataframe
        """        


        dataframes = list()

        index = 0
        for emgLabel in emgLabels:
            context = emgContexts[index]
            muscle = emgMuscles[index]
            dataframes.append(self.__getAmplitudebyPhase(analysisInstance,emgLabel+"_Rectify_Env_Norm",muscle,context))
            index+=1

        return pd.concat(dataframes)


    def __construcPandasSerie(self,emgLabel:str,muscle:str,context:str, cycleIndex:int,phase:str,
                              value:float,procedure:str)->pd.Series:
        """construct a pandas series

        Args:
            emgLabel (str): emg channel
            muscle (str): muscle name
            context (str): event context_
            cycleIndex (int): cycle index
            phase (str): movement phase ( e.g doubleStance)
            value (float): value
            procedure (str): name of the amplitude procedure

        Returns:
            pd.Series: _description_
        """
        iDict = OrderedDict([('ChannelLabel', emgLabel),
                     ('Label', muscle),
                     ('EventContext', context),
                     ('Cycle', cycleIndex),
                     ('Phase', phase),
                     ('Procedure', procedure),
                     ('Value', value)])
        return pd.Series(iDict)

    def __getAmplitudebyPhase(self,analysisInstance:Analysis,emglabel:str,muscle:str,context:str)->pd.DataFrame:
        """compute amplitude value for each phase

        Args:
            analysisInstance (Analysis): analysis instance
            emglabel (str): emg channel
            muscle (str): muscle name
            context (str): event context

        Returns:
            pd.DataFrame: populated dataframe 
        """
        procedure = "Amplitude"
        muscle = context[0]+muscle


        stanceValues =   analysisInstance.emgStats.pst['stancePhase', context]['values']
        doubleStance1Values =   analysisInstance.emgStats.pst['doubleStance1', context]['values']
        doubleStance2Values =   stanceValues - analysisInstance.emgStats.pst['doubleStance2', context]['values']

        normalizedCycleValues = analysisInstance.emgStats.data [emglabel,context]

        series = list()

        for i in range(0,len(normalizedCycleValues["values"])):

            # cycle
            lim0 = 0; lim1 = 100
            res_cycle=np.trapz(normalizedCycleValues["values"][i][lim0:lim1+1],
                         x=np.arange(lim0,lim1+1),axis=0)[0]

            serie = self.__construcPandasSerie(emglabel,muscle,context,
                                               int(i),
                                               "cycle",
                                               res_cycle,
                                               procedure)
            series.append(serie)

            # stance
            lim0 = 0; lim1 = int(stanceValues[i])
            res_stance=np.trapz(normalizedCycleValues["values"][i][lim0:lim1+1],
                         x=np.arange(lim0,lim1+1),axis=0)[0]

            serie = self.__construcPandasSerie(emglabel,muscle,context,
                                               int(i),
                                               "stance",
                                               res_stance,
                                               procedure)
            series.append(serie)


            # swing
            lim0 = int(stanceValues[i]); lim1 = 100
            res_swing=np.trapz(normalizedCycleValues["values"][i][lim0:lim1+1],
                         x=np.arange(lim0,lim1+1),axis=0)[0]

            serie = self.__construcPandasSerie(emglabel,muscle,context,
                                               int(i),
                                               "swing",
                                               res_swing,
                                               procedure)
            series.append(serie)


            # doubleStance1
            lim0 = 0; lim1 = int(doubleStance1Values[i])
            res_d1=np.trapz(normalizedCycleValues["values"][i][lim0:lim1+1],
                         x=np.arange(lim0,lim1+1),axis=0)[0]

            serie = self.__construcPandasSerie(emglabel,muscle,context,
                                               int(i),
                                               "doubleStance1",
                                               res_d1,
                                               procedure)
            series.append(serie)

            # double stance2
            lim0 = int(doubleStance2Values[i]); lim1 = int(stanceValues[i])
            res_d2=np.trapz(normalizedCycleValues["values"][i][lim0:lim1+1],
                         x=np.arange(lim0,lim1+1),axis=0)[0]

            serie = self.__construcPandasSerie(emglabel,muscle,context,
                                               int(i),
                                               "doubleStance2",
                                               res_d2,
                                               procedure)
            series.append(serie)

            lim0 = int(doubleStance1Values[i]); lim1 = int(doubleStance2Values[i])
            res_single=np.trapz(normalizedCycleValues["values"][i][lim0:lim1+1],
                         x=np.arange(lim0,lim1+1),axis=0)[0]

            serie = self.__construcPandasSerie(emglabel,muscle,context,
                                               int(i),
                                               "singleStance",
                                               res_single,
                                               procedure)
            series.append(serie)


        return pd.DataFrame(series)
