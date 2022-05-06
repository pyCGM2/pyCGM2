# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Configurator
#APIDOC["Draft"]=False
#--end--


import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2 import enums
from pyCGM2.Model import modelFilters
from pyCGM2.Model import modelDecorator


class argsManager_cgm(object):
    """ class for managing input arguments of all CGMi

    Args:
        settings (str): content of the CGM setting file
        args (argparse): argparse instance

    """


    def __init__(self, settings, args):
        self.settings = settings
        self.args = args


    def getHeadFlat(self):
        """ return value of  the head flat option"""
        if self.args.headFlat is not None:
            LOGGER.logger.warning("Head flat option : %s"%(str(bool(self.args.headFlat))))
            return  bool(self.args.headFlat)
        else:
            return bool(self.settings["Calibration"]["Head flat"])



    def getLeftFlatFoot(self):
        """ return value of  the left flat foot option"""
        if self.args.leftFlatFoot is not None:
            LOGGER.logger.warning("Left flat foot forces : %s"%(str(bool(self.args.leftFlatFoot))))
            return  bool(self.args.leftFlatFoot)
        else:
            return bool(self.settings["Calibration"]["Left flat foot"])

    def getRightFlatFoot(self):
        """ return value of  the right flat foot option"""
        if self.args.rightFlatFoot is not None:
            LOGGER.logger.warning("Right flat foot forces : %s"%(str(bool(self.args.rightFlatFoot))))
            return bool(self.args.rightFlatFoot)
        else:
            return  bool(self.settings["Calibration"]["Right flat foot"])


    def getMarkerDiameter(self):
        """ return marker diameter"""
        if self.args.markerDiameter is not None:
            LOGGER.logger.warning("marker diameter forced : %s", str(float(self.args.markerDiameter)))
            return float(self.args.markerDiameter)
        else:
            return float(self.settings["Global"]["Marker diameter"])

    def getIkAccuracy(self):
        """ return the IK accuracy"""
        if self.args.accuracy is not None:
            LOGGER.logger.warning("ik accuracy forced : %s", str(float(self.args.accuracy)))
            return float(self.args.accuracy)
        else:
            return float(self.settings["Global"]["IkAccuracy"])

    def getPointSuffix(self,checkValue):
        """ return point suffix

        Args:
            checkValue (str): string used if `check` is an enable attribute
            of the argparse instance

        """

        if hasattr(self.args,"check") and self.args.check:
            return checkValue
        else:
            if self.args.pointSuffix is not None:
                return self.args.pointSuffix
            else:
                if self.settings["Global"]["Point suffix"] =="None":
                    self.settings["Global"]["Point suffix"] = None

                return self.settings["Global"]["Point suffix"]

    def enableIKflag(self):
        " enable or disable the inverse kinematic processing"
        if self.args.noIk:
            return False
        else:
            if self.settings["Global"]["EnableIK"]:
                return True
            else:
                return False

    def getMomentProjection(self):
        """ return referentiel used for project joint moments"""
        if self.args.proj is not None:
            if self.args.proj == "Distal":
                return  enums.MomentProjection.Distal
            elif self.args.proj == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.args.proj == "Global":
                return  enums.MomentProjection.Global
            elif self.args.proj == "JCS":
                return enums.MomentProjection.JCS
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

        else:
            if self.settings["Fitting"]["Moment Projection"] == "Distal":
                return  enums.MomentProjection.Distal
            elif self.settings["Fitting"]["Moment Projection"] == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.settings["Fitting"]["Moment Projection"] == "Global":
                return  enums.MomentProjection.Global
            elif self.settings["Fitting"]["Moment Projection"] == "JCS":
                return enums.MomentProjection.JCS

            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

    def getManualForcePlateAssign(self):
        """ return foot side in contact with each force plate"""
        return self.args.mfpa

    def getIkWeightFile(self):
        """ return marker weight set used for the inverse kinematics"""
        return self.args.ikWeightFile

    def forceHjc(self,side):
        """ force HJC to local position expressed in the pelvic coordinate system"""
        if side == "left":
            if self.args.forceLHJC is not None:
                if len(self.args.forceLHJC) == 3:
                    lhjc = [float(i) for i in self.args.forceLHJC]
                else:
                    raise Exception("[pyCGM2] : left hjc position must have 3 components")
                return lhjc
            else:
                return None

        if side == "right":
            if self.args.forceRHJC is not None:
                if len(self.args.forceRHJC) == 3:
                    rhjc = [float(i) for i in self.args.forceRHJC]
                else:
                    raise Exception("[pyCGM2] : right hjc position must have 3 components")
                return rhjc
            else:
                return None

    def getAnalysisTitle(self):
        """ return the analysis title"""
        return self.args.analysisTitle


class argsManager_cgm1(argsManager_cgm):
    """ class for managing input arguments specific to the CGM1

    Args:
        settings (str): content of the CGM setting file
        args (argparse): argparse instance

    """
    def __init__(self, settings, args):
        super(argsManager_cgm1, self).__init__(settings, args)

    def getMomentProjection(self):
        """ return referentiel used for project joint moments"""

        if self.args.proj is not None:
            if self.args.proj == "Distal":
                return  enums.MomentProjection.Distal
            elif self.args.proj == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.args.proj == "Global":
                return  enums.MomentProjection.Global
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

        else:
            if self.settings["Fitting"]["Moment Projection"] == "Distal":
                return  enums.MomentProjection.Distal
            elif self.settings["Fitting"]["Moment Projection"] == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.settings["Fitting"]["Moment Projection"] == "Global":
                return  enums.MomentProjection.Global

            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")
