# -*- coding: utf-8 -*-
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Lib.Processing import progression
from pyCGM2.Tools import btkTools
from pyCGM2.Tools import opensimTools
import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional,Union

try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")



def prepareC3dFiles(DATA_PATH:str, staticFilename=None, dynamicData:List[Tuple[str, Optional[str]]]=None):
    """
    Prepares C3D files for OpenSim processing.

    This function processes static and dynamic C3D files for use with OpenSim. It converts C3D files to TRC format 
    and prepares ground reaction force (GRF) data for dynamic trials. For dynamic trials, the function can also 
    handle the assignment of force plates.

    Args:
        DATA_PATH (str): Path to the directory containing C3D files.

    Keyword Args:
        staticFilename (Optional[str]): Filename of the static trial. If provided, the file is converted to TRC format.
        dynamicData (Optional[List[Tuple[str, Optional[str]]]]): A list of tuples, each containing the filename of a dynamic trial and 
            an optional string specifying force plate assignments. Each character in the string corresponds to a force plate (e.g., 'XRL'). 
            'None' or an empty string indicates automatic detection.

    Example:
        >>> prepareC3dFiles(
                "c:/DATA/",
                staticFilename="myStatic.c3d",
                dynamicData=[["gait1.c3d", None], ["gait2.c3d", "XRL"]]
            )

    Note:
        In the example, 'gait1.c3d' will have automatic force plate assignment, while 'gait2.c3d' will have 
        'None' for force plate #1, 'Right' for #2, and 'Left' for #3. The number of characters in the string 
        should match the number of force plates.
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

