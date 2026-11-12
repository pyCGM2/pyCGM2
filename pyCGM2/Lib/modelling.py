"""High-level functions building generic models."""
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
        """Create and calibrate a single rigid segment model from markers.

        The technical frame of the segment, named ``Body``, is defined from the first
        three calibration markers (sequence XYZ), with the fourth one as origin.

        Args:
            acq (btk.btkAcquisition): Acquisition used for the calibration.
            calibration_markers (List[str]): Calibration markers (at least four).
            tracking_markers (List[str]): Tracking markers.
            calibFramesOfInterest (Optional[List[int]], optional): First and last frames used for
                the calibration. Defaults to None (all frames).

        Returns:
            pyCGM2.Model.model.Segment: The calibrated segment.
        """

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