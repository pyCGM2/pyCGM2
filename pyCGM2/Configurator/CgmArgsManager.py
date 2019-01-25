# -*- coding: utf-8 -*-
import numpy as np
import logging
from pyCGM2 import enums
from pyCGM2.Model import modelFilters, modelDecorator


class argsManager_cgm(object):
    def __init__(self, settings, args):
        self.settings = settings
        self.args = args

    def getLeftFlatFoot(self):
        if self.args.leftFlatFoot is not None:
            logging.warning("Left flat foot forces : %s"%(str(bool(self.args.leftFlatFoot))))
            return  bool(self.args.leftFlatFoot)
        else:
            return bool(self.settings["Calibration"]["Left flat foot"])

    def getRightFlatFoot(self):
        if self.args.rightFlatFoot is not None:
            logging.warning("Right flat foot forces : %s"%(str(bool(self.args.rightFlatFoot))))
            return bool(self.args.rightFlatFoot)
        else:
            return  bool(self.settings["Calibration"]["Right flat foot"])


    def getMarkerDiameter(self):
        if self.args.markerDiameter is not None:
            logging.warning("marker diameter forced : %s", str(float(self.args.markerDiameter)))
            return float(self.args.markerDiameter)
        else:
            return float(self.settings["Global"]["Marker diameter"])

    def getPointSuffix(self,checkValue):
        if self.args.check:
            return checkValue
        else:
            if self.args.pointSuffix is not None:
                return self.args.pointSuffix
            else:
                if self.settings["Global"]["Point suffix"] =="None":
                    self.settings["Global"]["Point suffix"] = None

                return self.settings["Global"]["Point suffix"]

    def enableIKflag(self):
        if self.args.noIk:
            return False
        else:
            if self.settings["Global"]["EnableIK"]:
                return True
            else:
                return False

    def getMomentProjection(self):
        if self.args.proj is not None:
            if self.args.proj == "Distal":
                return  enums.MomentProjection.Distal
            elif self.args.proj == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.args.proj == "Global":
                return  enums.MomentProjection.Global
            elif args.proj == "JCS":
                return pyCGM2Enums.MomentProjection.JCS
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
        return self.args.mfpa

    def getIkWeightFile(self):
        return self.args.ikWeightFile

    def forceHjc(self,side):
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


class argsManager_cgm1(argsManager_cgm):
    def __init__(self, settings, args):
        super(argsManager_cgm1, self).__init__(settings, args)

    def getMomentProjection(self):
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
