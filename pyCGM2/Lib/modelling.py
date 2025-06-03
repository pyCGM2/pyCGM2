import btk
import pyCGM2; LOGGER = pyCGM2.LOGGER
from typing import List, Tuple, Dict, Optional,Union

# pyCGM2 libraries
from pyCGM2.Model import model
from pyCGM2.Model import modelFilters


from pyCGM2 import enums




def createSingleSegment(acq:btk.btkAcquisition,
                        calibration_markers:List[str],
                        tracking_markers:List[str],
                        calibFramesOfInterest:Optional[List[str]]=None):

        segModel=model.Model()
        segModel.addSegment("Body",0,enums.SegmentSide.Central,
                       calibration_markers=calibration_markers, 
                       tracking_markers=tracking_markers)


        gcp=modelFilters.GeneralCalibrationProcedure()
        gcp.setDefinition('Body',
                          "TF",
                          sequence="XYZ",
                          pointLabel1=calibration_markers[0],
                          pointLabel2=calibration_markers[1],
                          pointLabel3=calibration_markers[2],
                          pointLabelOrigin=calibration_markers[3])

        modCal=modelFilters.ModelCalibrationFilter(gcp,acq,segModel)
        if calibFramesOfInterest is not None:
            modCal.setFrames(calibFramesOfInterest[0],calibFramesOfInterest[1])
    
        modCal.compute()

        # # Motion FILTER
        # modMotion=modelFilters.ModelMotionFilter(gcp,acq,segModel,enums.motionMethod.Sodervisk)
        # modMotion.compute()



        return segModel.getSegment("Body")