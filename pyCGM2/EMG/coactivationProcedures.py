""" This module contains procedures to compute co-activation indexes

check out the script : *\Tests\test_EMG.py* for examples


"""
import numpy as np
from typing import List, Tuple, Dict, Optional

class CoActivationProcedure(object):
    """Base class for co-activation procedures.

    This class serves as a foundation for specific co-activation index computation methods. 
    It should be extended to add specific functionalities for different types of co-activation index computations.
    """
    def __init__(self):
        """Initializes the CoActivationProcedure class."""
        pass

class UnithanCoActivationProcedure(CoActivationProcedure):
    """
    Co-activation procedure according to Unithan et al., 1996.

    This class implements the co-activation index computation as described by Unnithan et al. in their 1996 study on co-contraction and phasic activity during gait in children with cerebral palsy.

    Reference:
        Unnithan VB, Dowling JJ, Frost G, Volpe Ayub B, Bar-Or O. Cocontraction and phasic activity during GAIT in children with cerebral palsy. Electromyogr Clin Neurophysiol. 1996;36:487–494.
    """

    def __init__(self):
        """Initializes the UnithanCoActivationProcedure class."""
        super(UnithanCoActivationProcedure, self).__init__()


    def run(self, emg1:str, emg2:str):
        """Run the Unithan co-activation index computation procedure.

        Args:
            emg1 (str): EMG label of the first muscle.
            emg2 (str): EMG label of the second muscle.

        Returns:
            list: A list containing the computed co-activation index values.
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
    Co-activation index computation according to Falconer and Winter.

    This class implements the co-activation index computation as described by Falconer and Winter. The method is used to quantitatively assess co-contraction at the ankle joint during walking.

    Reference:
        Falconer K, Winter DA. Quantitative assessment of cocontraction at the ankle joint in walking. Electromyogr Clin Neurophysiol. 1985;25:135–149.
    """

    def __init__(self):
        """Initializes the FalconerCoActivationProcedure class."""
        super(FalconerCoActivationProcedure, self).__init__()


    def run(self, emg1:str, emg2:str):
        """Run the Falconer co-activation index computation procedure.

        Args:
            emg1 (str): EMG label of the first muscle.
            emg2 (str): EMG label of the second muscle.

        Returns:
            list: A list containing the computed co-activation index values.
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
