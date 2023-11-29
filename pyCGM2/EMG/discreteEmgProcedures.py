
"""
Module contains procedure for extracting discrete value ( e.g : amplitude) from each emg signal
"""


import numpy as np
import pandas as pd

from collections import OrderedDict
from pyCGM2.Processing import exporter
from pyCGM2.Processing.analysis import Analysis

from typing import List, Tuple, Dict, Optional

class DiscreteEmgProcedure(object):
    """Base class for procedures extracting discrete values from EMG signals.

    This class serves as a foundation for specific EMG data analysis procedures.
    It should be extended to implement methods for extracting specific discrete values such as amplitudes or other metrics from EMG signals.
    """
    def __init__(self):
        """Initializes the DiscreteEmgProcedure class."""
        pass

class AmplitudesProcedure(DiscreteEmgProcedure):
    """
    Procedure to compute EMG amplitudes for each gait phase.

    This class extends DiscreteEmgProcedure to specifically calculate the amplitude of EMG signals during different phases of gait. The amplitudes are calculated for each provided EMG channel and muscle.

    Attributes:
        NAME (str): Descriptive name of the procedure.
    """

    NAME = "EMG Amplitude from Integration for each gait phase"

    def __init__(self):
        """Initializes the AmplitudesProcedure class."""
        super(AmplitudesProcedure, self).__init__()

    def detect(self, analysisInstance:Analysis, emgLabels:List[str], emgMuscles:List[str], emgContexts:List[str])->pd.DataFrame:
        """Compute amplitudes for each gait phase.

        Args:
            analysisInstance (Analysis): An instance of the Analysis class containing EMG and gait data.
            emgLabels (List[str]): List of EMG channel labels.
            emgMuscles (List[str]): List of corresponding muscle names.
            emgContexts (List[str]): List of event contexts associated with each muscle.

        Returns:
            pd.DataFrame: A DataFrame containing computed amplitude values for each muscle, context, and gait phase.
        """        


        dataframes = []

        index = 0
        for emgLabel in emgLabels:
            context = emgContexts[index]
            muscle = emgMuscles[index]
            dataframes.append(self.__getAmplitudebyPhase(analysisInstance,emgLabel+"_Rectify_Env_Norm",muscle,context))
            index+=1

        return pd.concat(dataframes)


    def __construcPandasSerie(self,emgLabel:str,muscle:str,context:str, cycleIndex:int,phase:str,
                              value:float,procedure:str)->pd.Series:
        """Construct a Pandas Series for a single amplitude value.

        Args:
            emgLabel (str): EMG channel label.
            muscle (str): Muscle name.
            context (str): Event context.
            cycleIndex (int): Index of the gait cycle.
            phase (str): Name of the gait phase (e.g., 'doubleStance').
            value (float): Computed amplitude value.
            procedure (str): Name of the amplitude computation procedure.

        Returns:
            pd.Series: A Pandas Series object representing a single row of amplitude data.
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
        """Compute amplitude values for each phase of the gait cycle.

        Args:
            analysisInstance (Analysis): Analysis instance containing EMG data.
            emgLabel (str): EMG channel label.
            muscle (str): Muscle name.
            context (str): Event context.

        Returns:
            pd.DataFrame: A DataFrame populated with amplitude data for each phase of the gait cycle.
        """
        procedure = "Amplitude"
        muscle = context[0]+muscle


        stanceValues =   analysisInstance.emgStats.pst['stancePhase', context]['values']
        doubleStance1Values =   analysisInstance.emgStats.pst['doubleStance1', context]['values']
        doubleStance2Values =   stanceValues - analysisInstance.emgStats.pst['doubleStance2', context]['values']

        normalizedCycleValues = analysisInstance.emgStats.data [emglabel,context]

        series = []

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
