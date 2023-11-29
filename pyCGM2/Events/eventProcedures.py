"""
The module contains procedures for detecting foot contact event.

check out the script : *\Tests\test_events.py* for examples
"""

from typing import List, Tuple, Dict, Optional,Union
import pyCGM2; LOGGER = pyCGM2.LOGGER

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

from pyCGM2.Tools import  btkTools
from pyCGM2.Signal import detect_peaks
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures


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
