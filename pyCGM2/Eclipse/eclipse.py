# -*- coding: utf-8 -*-
import logging
import os
import configparser

import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

from pyCGM2.Model.CGM2 import forceplates
from pyCGM2.Tools import btkTools


def enfForcePlateAssignment(c3dFullFilename,mappedForcePlate):
    """
        Add Force plate assignement in the enf file

        :Parameters:
            - `c3dFullFilename` (str) - filename with path of the c3d
    """

    acqGait = btkTools.smartReader(str(c3dFullFilename))
    enfFile = str(c3dFullFilename[:-4]+".Trial.enf")

    if not os.path.isfile(enfFile):
        raise Exception ("[pyCGM2] - No enf file associated with the c3d")
    else:                
        # --------------------Modify ENF --------------------------------------
        configEnf = configparser.ConfigParser()
        configEnf.optionxform = str
        configEnf.read(enfFile)


        indexFP=1
        for letter in mappedForcePlate:

            if letter =="L": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Left"
            if letter =="R": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Right"
            if letter =="X": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Invalid"

            indexFP+=1

        tmpFile =str(c3dFullFilename[:-4]+".Trial.enf-tmp")
        with open(tmpFile, 'w') as configfile:
            configEnf.write(configfile)

        os.remove(enfFile)
        os.rename(tmpFile,enfFile)
        logging.warning("Enf file updated with Force plate assignment")
