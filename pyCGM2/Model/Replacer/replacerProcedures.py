import btk
import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2 libraries
from pyCGM2.Model import model
from pyCGM2.Model import modelFilters
from pyCGM2.Tools import btkTools

from pyCGM2 import enums



        # dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
        # dictRef["Left Thigh"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LKNE","LTHAP","LTHI","LKNE"]} }
        # dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RKNE","RTHAP","RTHI","RKNE"]} }
        # dictRef["Left Shank"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LANK","LTIAP","LTIB","LANK"]} }
        # dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RANK","RTIAP","RTIB","RANK"]} }

        # dictRef["Left Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LTOE","LAJC",None,"LAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis
        # dictRef["Right Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RTOE","RAJC",None,"RAJC"]} } # uncorrected Foot - use shank flexion axis (Y) as second axis


class ReplacerProcedure(object):
    """
    Base class for the pointer procedures
    """
    def __init__(self):
        pass


class ReplacerGeneralProcedure(ReplacerProcedure):

    def __init__(self,calibration_markers,tracking_markers,sequence):
        super(ReplacerGeneralProcedure, self).__init__()
        self.m_calibration_markers = calibration_markers
        self.m_trackingMarkers = tracking_markers
        self.m_sequence = sequence


    def compute(self,acq:btk.btkAcquisition):

        segModel=model.Model()
        segModel.addSegment("Body",0,enums.SegmentSide.Central,
                       calibration_markers=self.m_calibration_markers, 
                       tracking_markers = self.m_trackingMarkers)


        gcp=modelFilters.GeneralCalibrationProcedure()
        gcp.setDefinition('Body',
                          "TF",
                          sequence=self.m_sequence,
                          pointLabel1=self.m_calibration_markers[0],
                          pointLabel2=self.m_calibration_markers[1],
                          pointLabel3=self.m_calibration_markers[2],
                          pointLabelOrigin=self.m_calibration_markers[3])

        modCal=modelFilters.ModelCalibrationFilter(gcp,acq,segModel)
        modCal.compute()

        return segModel


class ReplacerSpecificProcedure(ReplacerProcedure):

    def __init__(self,scp,segment,technicalFrameLabel):
        super(ReplacerSpecificProcedure, self).__init__()
        
        self.definition = scp[segment][technicalFrameLabel]

        self.m_calibration_markers = self.definition["labels"]
        self.m_trackingMarkers = self.definition["labels"]
        self.m_sequence = self.definition["sequence"]


    def compute(self,acq:btk.btkAcquisition):

        segModel=model.Model()
        segModel.addSegment("Body",0,enums.SegmentSide.Central,
                       calibration_markers=self.m_calibration_markers, 
                       tracking_markers = self.m_trackingMarkers)


        gcp=modelFilters.GeneralCalibrationProcedure()
        gcp.setDefinition('Body',
                          "TF",
                          sequence=self.m_sequence,
                          pointLabel1=self.m_calibration_markers[0],
                          pointLabel2=self.m_calibration_markers[1],
                          pointLabel3=self.m_calibration_markers[2],
                          pointLabelOrigin=self.m_calibration_markers[3])

        modCal=modelFilters.ModelCalibrationFilter(gcp,acq,segModel)
        modCal.compute()

        return segModel


        