# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/EMG
#APIDOC["Draft"]=False
#--end--

""" This module contains co-activation procedures

check out the script : *\Tests\test_EMG.py* for examples


"""
import numpy as np


class UnithanCoActivationProcedure(object):
    """
    Coactivation procedure according Unithan et al, 1996

    Unnithan VB, Dowling JJ, Frost G, Volpe Ayub B, Bar-Or O. Cocontraction and phasic activity during GAIT in children with cerebral palsy. Electromyogr Clin Neurophysiol. 1996;36:487–494.

    """

    def __init__(self):
        pass

    def run(self, emg1, emg2):
        """run the procedure.

        Args:
            emg1 (str):  emg label .
            emg2 (str): emg label

        Returns:
            list: Coactivation index


        """

        out = list()
        for c1, c2 in zip(emg1, emg2):  # iterate along column
            commonEmg = np.zeros(((101, 1)))
            for i in range(0, 101):
                commonEmg[i, :] = np.minimum(c1[i], c2[i])
            res = np.trapz(commonEmg, x=np.arange(0, 101), axis=0)[0]
            out.append(res)

        return out


class FalconerCoActivationProcedure(object):
    """
    Coactivation index according falconer and Winter

    Falconer K, Winter DA. Quantitative assessment of cocontraction at the ankle joint in walking. Electromyogr Clin Neurophysiol. 1985;25:135–149. [PubMed] [Google Scholar]

    """

    def __init__(self):
        pass

    def run(self, emg1, emg2):
        """run the procedure.

        Args:
            emg1 (str):  emg label .
            emg2 (str): emg label

        Returns:
            list: Coactivation index


        """

        out = list()
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
