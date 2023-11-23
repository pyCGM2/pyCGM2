
# -*- coding: utf-8 -*-
import pyCGM2; LOGGER = pyCGM2.LOGGER
import os

import pyCGM2
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2 import enums

from pyCGM2.Lib.CGM import cgm2_1


DATA_PATH =  os.getcwd()+"//"

# chargement du fichier settings. Par defaut, il va chercher dans pyCGM2/Settings
# tu peux faire un copier coller de ce settings et le mettre dans ton repertoire)
settings = files.loadModelSettings(DATA_PATH,"CGM2_1-pyCGM2.settings")

# CALIBRATION--------------------------------
staticFile ="03367_05136_20200604-SBNNN-VDEF-02.c3d"
acqStatic = btkTools.smartReader(DATA_PATH+staticFile)

# choix des options
leftFlatFoot = settings["Calibration"]["Left flat foot"]
rightFlatFoot= settings["Calibration"]["Right flat foot"]
headFlat= settings["Calibration"]["Head flat"]
translators = settings["Translators"]
markerDiameter = settings["Global"]["Marker diameter"]
HJC = settings["Calibration"]["HJC"]
pointSuffix = settings["Global"]["Point suffix"]

# definitions des mp a partir des metadata
required_mp = {}
required_mp["Bodymass"] = 75.0
required_mp["Height"]= 1750
required_mp["LeftLegLength"] = 800
required_mp["LeftKneeWidth"] = 90
required_mp["RightLegLength"] = 800
required_mp["RightKneeWidth"] = 90
required_mp["LeftAnkleWidth"] = 60
required_mp["RightAnkleWidth"] = 60
required_mp["LeftSoleDelta"] = 0
required_mp["RightSoleDelta"] = 0
required_mp["LeftShoulderOffset"] = 0
required_mp["LeftElbowWidth"] = 0
required_mp["LeftWristWidth"] = 0
required_mp["LeftHandThickness"] = 0
required_mp["RightShoulderOffset"] = 0
required_mp["RightElbowWidth"] = 0
required_mp["RightWristWidth"] = 0
required_mp["RightHandThickness"]= 0

optional_mp = {}
optional_mp["InterAsisDistance"]= 0
optional_mp["LeftAsisTrocanterDistance"]= 0
optional_mp["LeftTibialTorsion"]= 0
optional_mp["LeftThighRotation"]= 0
optional_mp["LeftShankRotation"]= 0
optional_mp["RightAsisTrocanterDistance"]= 0
optional_mp["RightTibialTorsion"]= 0
optional_mp["RightThighRotation"]= 0
optional_mp["RightShankRotation"]= 0


model,finalAcqStatic,error = cgm2_1.calibrate(DATA_PATH,
    staticFile,
    translators,
    required_mp,
    optional_mp,
    leftFlatFoot,
    rightFlatFoot,
    headFlat,
    markerDiameter,
    HJC,
    pointSuffix)

# FITTING ----------------------------------------------------------
trialName = "03367_05136_20200604-GBNNN-VDEF-01.c3d"

momentProjection = enums.enumFromtext(settings["Fitting"]["Moment Projection"],enums.MomentProjection) 
pointSuffix = settings["Global"]["Point suffix"]


# force plate assignement a parir du c3d
acq = btkTools.smartReader(DATA_PATH+trialName)
mfpa = forceplates.matchingFootSideOnForceplate(acq)

acqGait,detectAnomaly = cgm2_1.fitting(model,DATA_PATH, trialName,
    translators,
    markerDiameter,
    pointSuffix,
    mfpa,
    momentProjection,
    frameInit= None, frameEnd= None )

btkTools.smartWriter(acqGait, DATA_PATH+trialName[:-4]+"-modelled.c3d")
