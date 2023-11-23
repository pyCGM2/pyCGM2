
"""
An inspector procedure just inspect the content of an acquisition. It doesn't alter it.

"""
import btk
from pyCGM2.Tools import btkTools
import pyCGM2; LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional,Union


class InspectorProcedure(object):
    def __init__(self):
        pass


class MarkerPresenceDetectionProcedure(InspectorProcedure):
    """Procedure to check marker presence in the acquisition.

    Args:
        markers (Optional[list], optional): marker names. Defaults to None.

    """
    def __init__(self,markers:Optional[List]=None):
        super(MarkerPresenceDetectionProcedure, self).__init__()

        self.markers = markers

    def run(self,acq:btk.btkAcquisition,filename:str,options:dict)-> dict:
        """run the procedure

        Args:
            acq (btk.btkAcquisition): an btk acquisition instance
            filename (str): c3d filename
            options (dict): options

        Returns:
            dict: present and missing markers
        """

        markersIn = []
        markersOut = []

        for marker in self.markers:
            try:
                acq.GetPoint(marker)
            except RuntimeError:
                markersOut.append(marker)
            else:
                if not btkTools.isPhantom(acq,marker):
                    markersOut.append(marker)
                else:
                    markersIn.append(marker)

        if markersIn !=[] and markersOut!=[]:
            for markerOut in markersOut:
                LOGGER.logger.info("[pyCGM2-Inspector]  marker [%s] - not exist in the file [%s]"%(markerOut, filename))

        return {"In":markersIn, "Out":markersOut}
