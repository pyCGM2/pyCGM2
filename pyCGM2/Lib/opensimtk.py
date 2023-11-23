# -*- coding: utf-8 -*-
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Lib.Processing import progression
from pyCGM2.Tools import btkTools
from pyCGM2.Tools import opensimTools
import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Optional, Tuple

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")



def prepareC3dFiles(DATA_PATH:str, staticFilename=None, dynamicData:List[Tuple[str, Optional[str]]]=None):
    """ prepare c3d data to opensim processingf.


    Args:
        DATA_PATH (str): path to your data

    Keyword Arguments:
        staticFilename (str) : static trial filename
        dynamicData (List[Tuple[str, Optional[str]]]): list of 2-elements lists composed of the dynamic filename and its assigned force plates

    Example:

    e.g : in the snippet below, there are 2 dynamic gait trials. the assigned force plate is detected for *gait1.c3d* automatically. 
    For *gait2.c3d*, the assigned force plates None for force plate #1, Right for fp#2 and Left for fp#3. 
    The number of capitals is the number of force plate. 

    .. code-block:: python

        prepareC3dFiles("c:\\DATA\\", 
            staticFilename="myStatic.c3d",
            dynamicData=[["gait1.c3d",None], ["gait2.c3d","XRL"]])   



    """
    if staticFilename is not None:
        acq = btkTools.smartReader(DATA_PATH+staticFilename)
        btkTools.smartWriter(acq, DATA_PATH + staticFilename[:-4], extension="trc")
        

    if dynamicData is not None:

        for it in  dynamicData:

            if len(it) == 1:
                dynamicFilename = it[0]
                mfpa = None

            if len(it) == 2:
                dynamicFilename = it[0]
                mfpa = it[1]

            acq = btkTools.smartReader(DATA_PATH+dynamicFilename)
            acqClone = btk.btkAcquisition.Clone(acq)

            progressionAxis, forwardProgression, globalFrame = progression.detectProgressionFrame(
                acqClone)

            # affect rotation Lab/Osim
            opensimTools.transformMarker_ToOsimReferencial(acqClone, progressionAxis, forwardProgression)

            # export trc
            btkTools.smartWriter(acqClone, DATA_PATH + dynamicFilename[:-4], extension="trc")

            # generate grf
            opensimTools.footReactionMotFile(
                    acq, DATA_PATH+dynamicFilename[:-4]+"_grf.mot",
                    progressionAxis, forwardProgression, mfpa=mfpa)

# def def driveOsim(DATA_PATH, staticFilename=None, dynamicData=None):
#     scaledOsimName = "LOISEAU Matys Cal 01-CGM23-ScaledModel.osim"
    
#     # -- angle poplité
#     procAnaDriven = opensimAnalysesInterfaceProcedure.AnalysesXmlCgmDrivenModelProcedure(DATA_PATH,scaledOsimName,"musculoskeletal_modelling/Poplity","CGM2.3")
#     procAnaDriven.setPose("Poplity",{"hip_flexion_r":90, "hip_flexion_l":90,
#                                     "knee_flexion_r":-50, "knee_flexion_l":-25 })
#     procAnaDriven.prepareXml()
#     oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAnaDriven)
#     oiamf.run()

