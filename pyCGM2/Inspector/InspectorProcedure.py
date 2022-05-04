#APIDOC: /Low level/Inspector
"""
Module contains inspector procedure.

An inspector just inspect the content of an acquisition. It doesn't alter it.

"""

from pyCGM2.Tools import btkTools
from pyCGM2.Signal import anomaly
from pyCGM2.Math import derivation
from pyCGM2.Signal import signal_processing
from pyCGM2.Utils import utils

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pyCGM2; LOGGER = pyCGM2.LOGGER


try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")



class AbstractInspectorProcedure(object):
    def __init__(self):
        pass

    def run(self,acq,filename,options):
        pass

class MarkerPresenceDetectionProcedure(AbstractInspectorProcedure):
    """Procedure to check marker presence in the acquisition.

    Args:
        markers (list,Optional[None]):marker labels

    """
    def __init__(self,markers=None):
        super(MarkerPresenceDetectionProcedure, self).__init__()

        self.markers = markers

    def run(self,acq,filename,options):

        markersIn = list()
        markersOut = list()

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
