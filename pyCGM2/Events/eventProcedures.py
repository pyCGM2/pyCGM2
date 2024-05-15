"""
The module contains procedures for detecting foot contact event.

check out the script : *\Tests\test_events.py* for examples
"""

from typing import List, Tuple, Dict, Optional,Union
import pyCGM2; LOGGER = pyCGM2.LOGGER

import btk

from pyCGM2.Tools import  btkTools
from pyCGM2.Signal import detect_peaks
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures
from pyCGM2.Math import derivation


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences


# --- abstract procedure
class EventProcedure(object):
    """
    Abstract class for event procedures.

    This class serves as a foundation for specific event detection procedures in gait analysis. 
    It should be extended to implement methods for detecting specific types of gait events.
    """
    def __init__(self):
        pass


#-------- EVENT PROCEDURES  ----------


class ZeniProcedure(EventProcedure):
    """
    Gait event detection procedure based on the method described by Zeni et al., 2008.

    This procedure implements a gait event detection algorithm that identifies foot strike and foot off events based on the motion of heel and toe markers relative to the pelvis.

    Attributes:
        description (str): Description of the event detection procedure.
        footStrikeOffset (int): Systematic offset applied to each foot strike event.
        footOffOffset (int): Systematic offset applied to each foot off event.
    """

    def __init__(self):
        """
        Initializes the ZeniProcedure class.
        """
        super(ZeniProcedure, self).__init__()
        self.description = "Zeni (2008)"
        self.footStrikeOffset = 0
        self.footOffOffset = 0

    def setFootStrikeOffset(self,value:int):
        """
        Set a systematic offset to each foot strike event.

        Args:
            value (int): Frame offset to apply to each foot strike event.
        """

        self.footStrikeOffset = value

    def setFootOffOffset(self,value:int):
        """
        Set a systematic offset to each foot off event.

        Args:
            value (int): Frame offset to apply to each foot off event.
        """
        self.footOffOffset = value

    def detect(self,acq:btk.btkAcquisition)-> Union[Tuple[int, int, int, int], int] :
        """
        Detect events using the Zeni method.

        Args:
            acq (btk.btkAcquisition): A BTK acquisition instance containing motion capture data.

        Returns:
            Union[Tuple[int, int, int, int], int]: Frames indicating the left foot strike, left foot off, 
                                                   right foot strike, and right foot off respectively. 
                                                   Returns 0 if detection fails.
        """


        ff=acq.GetFirstFrame()


        if btkTools.isPointsExist(acq,["LPSI","RPSI","LHEE","LTOE","RHEE","RTOE"]):
            pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
            pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]


            longAxisIndex = 0 if progressionAxis == "X" else 1

            sacrum=(acq.GetPoint("LPSI").GetValues() + acq.GetPoint("RPSI").GetValues()) / 2.0

            #Left
            heel_left = acq.GetPoint("LHEE").GetValues()
            toe_left = acq.GetPoint("LTOE").GetValues()

            diffHeel_left = heel_left-sacrum
            diffToe_left = toe_left-sacrum

            #Right
            heel_right = acq.GetPoint("RHEE").GetValues()
            toe_right = acq.GetPoint("RTOE").GetValues()

            diffHeel_right = heel_right-sacrum
            diffToe_right = toe_right-sacrum


            if  forwardProgression:
                indexes_fs_left = detect_peaks.detect_peaks(diffHeel_left[:,longAxisIndex])+ff
                indexes_fo_left = detect_peaks.detect_peaks(-diffToe_left[:,longAxisIndex])+ff
            else:
                indexes_fs_left = detect_peaks.detect_peaks(-diffHeel_left[:,longAxisIndex])+ff
                indexes_fo_left = detect_peaks.detect_peaks(diffToe_left[:,longAxisIndex])+ff

            if  forwardProgression:
                indexes_fs_right = detect_peaks.detect_peaks(diffHeel_right[:,longAxisIndex])+ff
                indexes_fo_right = detect_peaks.detect_peaks(-diffToe_right[:,longAxisIndex])+ff
            else:
                indexes_fs_right = detect_peaks.detect_peaks(-diffHeel_right[:,longAxisIndex])+ff
                indexes_fo_right = detect_peaks.detect_peaks(diffToe_right[:,longAxisIndex])+ff

            return indexes_fs_left+self.footStrikeOffset,indexes_fo_left+self.footOffOffset, indexes_fs_right+self.footStrikeOffset, indexes_fo_right+self.footOffOffset

        else:
            LOGGER.logger.error("[pyCGM2]: Zeni event detector impossible to run. Pelvic LPSI-RPSI or foot markers(HEE or TOE) are missing ")
            return 0


class OconnorProcedure(EventProcedure):
    """Detects initial contacts and final contacts from the heel and toe marker position data.

    See:
        O'Connor CM, et al. Gait Posture. 2007. doi: 10.1016/j.gaitpost.2006.05.016 

    Parameters
    ----------
    heel_pos, toe_pos : (N, D) ndarray, (N, D) ndarray
        The marker position data with N time steps across D channels,
        for the heel and toe marker, respectively.
    fs : int, float
        The sampling frequency (in Hz).

    Returns
    -------
    ix_IC, ix_FC : ndarray, ndarray
        The indexes corresponding to initial contacts and final contacts, respectively.
    """

    def __init__(self, check=False):
        """
        Initializes the OconnorProcedure class.
        """
        super(OconnorProcedure, self).__init__()
        self.description = "Oconnor (2007)"

        self.check = check


    def detect(self,acq:btk.btkAcquisition)-> Union[Tuple[int, int, int, int], int] :
        """
        Detect events using the Oconnor method.

        Args:
            acq (btk.btkAcquisition): A BTK acquisition instance containing motion capture data.

        Returns:
            Union[Tuple[int, int, int, int], int]: Frames indicating the left foot strike, left foot off, 
                                                   right foot strike, and right foot off respectively. 
                                                   Returns 0 if detection fails.
        """


        ff=acq.GetFirstFrame()
        freq=acq.GetPointFrequency()


        if btkTools.isPointsExist(acq,["LHEE","LTOE","RHEE","RTOE"]):
            pfp = progressionFrameProcedures.PointProgressionFrameProcedure()
            pff = progressionFrameFilters.ProgressionFrameFilter(acq,pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            globalFrame = pff.outputs["globalFrame"]
            forwardProgression = pff.outputs["forwardProgression"]


            longAxisIndex = 0 if progressionAxis == "X" else 1


            # Calculate virtual foot center 
            lfoot=(acq.GetPoint("LHEE").GetValues() + acq.GetPoint("LTOE").GetValues()) / 2.0
            rfoot=(acq.GetPoint("RHEE").GetValues() + acq.GetPoint("RTOE").GetValues()) / 2.0

            # Calculate the velocity
            lfootvelocity = derivation.splineFittingDerivation(lfoot,freq)
            rfootvelocity = derivation.splineFittingDerivation(rfoot,freq)


            # Left side
            # Detect peaks in the foot center forward velocity
            ix_pks_x, _ = find_peaks(lfootvelocity[:,longAxisIndex], distance=freq//4)
            pk_proms = peak_prominences(lfootvelocity[:,longAxisIndex], ix_pks_x)
            ix_pks_x = ix_pks_x[pk_proms[0] > 0.1*max(pk_proms[0])]

            # Detect negative peaks in the foot center vertical velocity
            ix_pks_z_neg, _ = find_peaks(-lfootvelocity[:,2], height=0)
            ix_pks_z_pos, _ = find_peaks(lfootvelocity[:,2])

            ix_IC, ix_FC = [], []
            for i in range(len(ix_pks_x)):
                f = np.argwhere(np.logical_and(ix_pks_z_neg > ix_pks_x[i], ix_pks_z_neg <= ix_pks_x[i]+(freq+freq//2)))[:,0]
                if len(f) > 0:
                    ix_IC.append(ix_pks_z_neg[f[0]])
                g = np.argwhere(np.logical_and(ix_pks_z_pos < ix_pks_x[i], ix_pks_z_pos >= ix_pks_x[i]-(freq+freq//2)))[:,0]
                if len(g) > 0:
                    ix_FC.append(ix_pks_z_pos[g[-1]])
            
            if self.check:

                fig, ax = plt.subplots(3, 1)

                heel_pos = acq.GetPoint("LHEE").GetValues()
                toe_pos = acq.GetPoint("LTOE").GetValues()


                ax[0].plot(heel_pos[:,2], ls='-', lw=1, c=(0, 0.5, 0, 0.4))
                ax[0].plot(toe_pos[:,2], ls='-.', lw=1, c=(0, 0.5, 0, 0.2))
                ax[0].plot(lfoot[:,2], ls='-', lw=1, c=(0, 0.5, 0))
                ax[0].set_xlim((0, lfoot.shape[0]))

                ax[1].plot(lfootvelocity[:,2], ls='-', lw=1, c=(0, 0.5, 0))
                ax[1].plot(ix_IC, lfootvelocity[ix_IC,2], 'o', mfc='none', mec=(0, 0.5, 0), ms=8)
                ax[1].plot(ix_FC, lfootvelocity[ix_FC,2], 's', mfc='none', mec=(0, 0.5, 0), ms=8)
                ax[1].set_xlim((0, lfootvelocity.shape[0]))
                
                ax[2].plot(lfootvelocity[:,0], ls='-', lw=1, c=(0, 0.5, 0))
                ax[2].plot(ix_pks_x, lfootvelocity[ix_pks_x,0], '^', mfc='none', mec=(0, 0.5, 0), ms=4)
                ax[2].set_xlim((0, lfootvelocity.shape[0]))
                # plt.show()

            ix_IC_L =  [it+ff for it in ix_IC]
            ix_FC_L =  [it+ff for it in ix_FC]


            # Right side
            # Detect peaks in the foot center forward velocity
            ix_pks_x, _ = find_peaks(rfootvelocity[:,longAxisIndex], distance=freq//4)
            pk_proms = peak_prominences(rfootvelocity[:,longAxisIndex], ix_pks_x)
            ix_pks_x = ix_pks_x[pk_proms[0] > 0.1*max(pk_proms[0])]

            # Detect negative peaks in the foot center vertical velocity
            ix_pks_z_neg, _ = find_peaks(-rfootvelocity[:,2], height=0)
            ix_pks_z_pos, _ = find_peaks(rfootvelocity[:,2])

            ix_IC, ix_FC = [], []
            for i in range(len(ix_pks_x)):
                f = np.argwhere(np.logical_and(ix_pks_z_neg > ix_pks_x[i], ix_pks_z_neg <= ix_pks_x[i]+(freq+freq//2)))[:,0]
                if len(f) > 0:
                    ix_IC.append(ix_pks_z_neg[f[0]])
                g = np.argwhere(np.logical_and(ix_pks_z_pos < ix_pks_x[i], ix_pks_z_pos >= ix_pks_x[i]-(freq+freq//2)))[:,0]
                if len(g) > 0:
                    ix_FC.append(ix_pks_z_pos[g[-1]])

            ix_IC_R =  [it+ff for it in ix_IC]
            ix_FC_R =  [it+ff for it in ix_FC]

            if self.check:

                fig, ax = plt.subplots(3, 1)

                heel_pos = acq.GetPoint("RHEE").GetValues()
                toe_pos = acq.GetPoint("RTOE").GetValues()


                ax[0].plot(heel_pos[:,2], ls='-', lw=1, c=(0, 0.5, 0, 0.4))
                ax[0].plot(toe_pos[:,2], ls='-.', lw=1, c=(0, 0.5, 0, 0.2))
                ax[0].plot(lfoot[:,2], ls='-', lw=1, c=(0, 0.5, 0))
                ax[0].set_xlim((0, lfoot.shape[0]))

                ax[1].plot(rfootvelocity[:,2], ls='-', lw=1, c=(0, 0.5, 0))
                ax[1].plot(ix_IC, rfootvelocity[ix_IC,2], 'o', mfc='none', mec=(0, 0.5, 0), ms=8)
                ax[1].plot(ix_FC, rfootvelocity[ix_FC,2], 's', mfc='none', mec=(0, 0.5, 0), ms=8)
                ax[1].set_xlim((0, rfootvelocity.shape[0]))
                
                ax[2].plot(rfootvelocity[:,0], ls='-', lw=1, c=(0, 0.5, 0))
                ax[2].plot(ix_pks_x, rfootvelocity[ix_pks_x,0], '^', mfc='none', mec=(0, 0.5, 0), ms=4)
                ax[2].set_xlim((0, rfootvelocity.shape[0]))
                
                plt.show()


            return ix_IC_L, ix_FC_L, ix_IC_R, ix_FC_R

        else:
            LOGGER.logger.error("[pyCGM2]: Oconnor event detector impossible to run.  foot markers(HEE or TOE) are missing ")
            return 0