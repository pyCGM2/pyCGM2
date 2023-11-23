""" This module contains procedures to compute co-activation indexes

check out the script : *\Tests\test_EMG.py* for examples


"""
import numpy as np
from typing import List, Tuple, Dict, Optional

class CoActivationProcedure(object):
    def __init__(self):
        pass

class UnithanCoActivationProcedure(CoActivationProcedure):
    """
    Coactivation procedure according Unithan et al, 1996

    Unnithan VB, Dowling JJ, Frost G, Volpe Ayub B, Bar-Or O. Cocontraction and phasic activity during GAIT in children with cerebral palsy. Electromyogr Clin Neurophysiol. 1996;36:487–494.

    """

    def __init__(self):
        super(UnithanCoActivationProcedure, self).__init__()


    def run(self, emg1:str, emg2:str):
        """run the procedure.

        Args:
            emg1 (str):  emg label .
            emg2 (str): emg label

        Returns:
            list: Coactivation index


        """

        out = []
        for c1, c2 in zip(emg1, emg2):  # iterate along column
            commonEmg = np.zeros(((101, 1)))
            for i in range(0, 101):
                commonEmg[i, :] = np.minimum(c1[i], c2[i])
            res = np.trapz(commonEmg, x=np.arange(0, 101), axis=0)[0]
            out.append(res)

        return out


class FalconerCoActivationProcedure(CoActivationProcedure):
    """
    Coactivation index according falconer and Winter

    Falconer K, Winter DA. Quantitative assessment of cocontraction at the ankle joint in walking. Electromyogr Clin Neurophysiol. 1985;25:135–149. [PubMed] [Google Scholar]

    """

    def __init__(self):
        super(FalconerCoActivationProcedure, self).__init__()


    def run(self, emg1:str, emg2:str):
        """run the procedure.

        Args:
            emg1 (str):  emg label .
            emg2 (str): emg label

        Returns:
            list: Coactivation index


        """

        out = []
        for c1, c2 in zip(emg1, emg2):  # iterate along column

            commonEmg = np.zeros(((101, 1)))
            sumEmg = np.zeros(((101, 1)))
            for i in range(0, 101):
                commonEmg[i, :] = np.minimum(c1[i], c2[i])
                sumEmg[i, :] = c1[i]+c2[i]

            areaNum = np.trapz(commonEmg, x=np.arange(0, 101), axis=0)[0]
            areaDen = np.trapz(sumEmg, x=np.arange(0, 101), axis=0)[0]
            res = 2.0 * areaNum / areaDen
            out.append(res)

        return out
