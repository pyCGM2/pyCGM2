# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_replacer.py::Test_replacer::test_PelvicReplacer

import pytest
import numpy as np
import pyCGM2

from pyCGM2.Model import  modelDecorator

from pyCGM2.Model.Replacer import replacerFilter
from pyCGM2.Model.Replacer import replacerProcedures
from pyCGM2.Tools import btkTools


import ipdb
import os
import matplotlib.pyplot as plt


import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2


from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools

from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model import  modelFilters,modelDecorator

from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Nexus import vskTools

from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures.scaling import opensimScalingInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.analysisReport import opensimAnalysesInterfaceProcedure


class Test_replacer:

    def test_PelvicGeneralReplacer(self):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\replacer\\cgm23\\"
        acqStatic =  btkTools.smartReader(MAIN_PATH+"static.c3d")


        valSACR=(acqStatic.GetPoint("LPSI").GetValues() + acqStatic.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acqStatic,"SACR",valSACR,desc="")
        valMidAsis=(acqStatic.GetPoint("LASI").GetValues() + acqStatic.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acqStatic,"midASIS",valMidAsis,desc="")


        proc = replacerProcedures.ReplacerGeneralProcedure(["RASI","LASI","SACR","midASIS"],["RASI","LASI","SACR","midASIS"],"YZX")

        globalPosition=np.array([0, 150.0, 1200.0])
        rf = replacerFilter.ReplacerFilter(proc,"noteTest",globalPosition)
        rf.run(acqStatic)
        localPosition = rf.getLocalPosition()


    def test_PelvicSpecificReplacer(self):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\replacer\\cgm23\\"
        acqStatic =  btkTools.smartReader(MAIN_PATH+"static.c3d")

        valSACR=(acqStatic.GetPoint("LPSI").GetValues() + acqStatic.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acqStatic,"SACR",valSACR,desc="")
        valMidAsis=(acqStatic.GetPoint("LASI").GetValues() + acqStatic.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acqStatic,"midASIS",valMidAsis,desc="")

        scp = modelFilters.StaticCalibrationProcedure(cgm2.CGM2_3())

        proc = replacerProcedures.ReplacerSpecProcedure(scp,"Pelvis","TF")
        globalPosition=np.array([0, 150.0, 1200.0])
        rf = replacerFilter.ReplacerFilter(proc,"noteTest",globalPosition)
        rf.run(acqStatic)
        localPosition = rf.getLocalPosition()

    

    def test_PelvicReplacerThenDecorated(self):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "LowLevel\\replacer\\cgm23\\"

        pointerFilename = "static.c3d"
        staticFilename = "static.c3d"

        acqPointer =  btkTools.smartReader(MAIN_PATH+pointerFilename)

        #-------REPLACER------
        #---------------------
        valSACR=(acqPointer.GetPoint("LPSI").GetValues() + acqPointer.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acqPointer,"SACR",valSACR,desc="")
        valMidAsis=(acqPointer.GetPoint("LASI").GetValues() + acqPointer.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(acqPointer,"midASIS",valMidAsis,desc="")


        proc = replacerProcedures.ReplacerGeneralProcedure(["RASI","LASI","SACR","midASIS"],["RASI","LASI","SACR","midASIS"],"YZX")

        globalPosition=np.array([0, 150.0, 1200.0])
        rf = replacerFilter.ReplacerFilter(proc,"noteTest",globalPosition)
        rf.run(acqPointer)
        localPosition = rf.getLocalPosition()
        





        #----------CGM23 CALIBRATION----------
        #-------------------------------------

        #----Appel des settings----
        modelVersion = "CGM2.3"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]

        markerDiameter=14


        #--- recuperation des données anthropometriques----

        # possibilité 1 - appel du fichier vsk 
        vsk = vskTools.Vsk(MAIN_PATH + "MRI-US-01.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        # --- Calibration du modele CGM2.3 ---

        acqStatic = btkTools.smartReader(MAIN_PATH +  staticFilename) # construction btkAcquisition instance

        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        # configuration du modele
        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model =cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)

        # filtre 1 ModelCalibrationFilter => calibration initiale du model  
        scp = modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # raffinement des centres articulaires 
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")


        # ajout des nodes replacer

        nodeLabel = "noteTest"
        model.getSegment("Pelvis").getReferential("TF").static.addNode(nodeLabel,
                                                                              localPosition,positionType="Local",desc = "fromReplacer")
        
        # filtre 2 ModelCalibrationFilter => update de la calibration de maniere a prendre en compte les nouveaux centres de rotation  
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        markerDiameter=markerDiameter).compute()
        
        
        print(model.getSegment("Pelvis").getReferential("opensim").static.getNode_byLabel(nodeLabel).getLocal())


