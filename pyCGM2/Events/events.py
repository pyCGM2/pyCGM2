# -*- coding: utf-8 -*-
import logging

try: 
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
from pyCGM2.Tools import  btkTools
from pyCGM2.Signal import detect_peaks
from pyCGM2.Processing import progressionFrame


#-------- EVENT PROCEDURES  ----------


# --- calibration procedure
class ZeniProcedure(object):
    """
        Gait Event detection from Zeni
    """

    def __init__(self):
        self.description = "Zeni (2008)"
        self.footStrikeOffset = 0
        self.footOffOffset = 0

    def setFootStrikeOffset(self,value):
        self.footStrikeOffset = value

    def setFootOffOffset(self,value):
        self.footOffOffset = value

    def detect(self,acq):
        """
        """
        ff=acq.GetFirstFrame()


        if btkTools.isPointsExist(acq,["LPSI","RPSI","LHEE","LTOE","RHEE","RTOE"]):
            pfp = progressionFrame.PelvisProgressionFrameProcedure()
            pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
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
            logging.error("[pyCGM2]: Zeni event detector impossible to run. Pelvic LPSI-RPSI or foot markers(HEE or TOE) are missing ")
            return 0

class EventFilter(object):
    """

    """
    def __init__(self,procedure,acq):
        """
            :Parameters:
        """

        self.m_aqui = acq
        self.m_procedure = procedure
        self.m_state = None

    def getState(self):
        return self.m_state

    def detect(self):
        """
            Run the motion filter
        """
        pf = self.m_aqui.GetPointFrequency()

        eventDescriptor = self.m_procedure.description

        if self.m_procedure.detect(self.m_aqui) == 0:
            self.m_state = False
        else:
            indexes_fs_left,indexes_fo_left,indexes_fs_right,indexes_fo_right =  self.m_procedure.detect(self.m_aqui)
            self.m_state = True
            for ind in indexes_fs_left:
                ev = btk.btkEvent('Foot Strike', (ind-1)/pf, 'Left', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetId(1)
                self.m_aqui.AppendEvent(ev)

            for ind in indexes_fo_left:
                ev = btk.btkEvent('Foot Off', (ind-1)/pf, 'Left', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetId(2)
                self.m_aqui.AppendEvent(ev)

            for ind in indexes_fs_right:
                ev = btk.btkEvent('Foot Strike', (ind-1)/pf, 'Right', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetId(1)
                self.m_aqui.AppendEvent(ev)

            for ind in indexes_fo_right:
                ev = btk.btkEvent('Foot Off', (ind-1)/pf, 'Right', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetId(2)
                self.m_aqui.AppendEvent(ev)
