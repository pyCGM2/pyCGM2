# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
import json
import numpy as np

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
pyCGM2.CONFIG.addNexusPythonSdk()
import ViconNexus


# openMA
pyCGM2.CONFIG.addOpenma()
import ma.io
import ma.body

#btk
pyCGM2.CONFIG.addBtk()
import btk



# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator

from pyCGM2 import viconInterface

def checkCGM1_StaticMarkerConfig(acqStatic):

    out = dict()

    # medial ankle markers
    out["leftMedialAnkleFlag"] = True if btkTools.isPointExist(acqStatic,"LMED") else False
    out["rightMedialAnkleFlag"] = True if btkTools.isPointExist(acqStatic,"RMED") else False

    # medial ankle markers
    out["leftMedialKneeFlag"] = True if btkTools.isPointExist(acqStatic,"LMEPI") else False
    out["rightMedialKneeFlag"] = True if btkTools.isPointExist(acqStatic,"RMEPI") else False


    # kad
    out["leftKadFlag"] = True if btkTools.isPointsExist(acqStatic,["LKAX","LKD1","LKD2"]) else False
    out["rightKadFlag"] = True if btkTools.isPointsExist(acqStatic,["RKAX","RKD1","RKD2"]) else False

    return out




if __name__ == "__main__":

    plt.close("all")
    DEBUG = False

    pyNEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

    print NEXUS_PYTHON_CONNECTED

    #NEXUS_PYTHON_CONNECTED = True

    if NEXUS_PYTHON_CONNECTED: # run Operation

        # inputs

        if DEBUG:
            DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\CGM1-Calibration\\"
            calibrateFilenameLabelledNoExt = "static Cal 01-noKAD-noAnkleMed" #"static Cal 01-noKAD-noAnkleMed" #
            pyNEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = pyNEXUS.GetTrialName()


        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)



        if not os.path.isfile( DATA_PATH + "pyCGM2.inputs"):
            raise Exception ("pyCGM2.inputs file doesn't exist")
        else:
            inputs = json.loads(open(DATA_PATH +'pyCGM2.inputs').read())

        flag_leftFlatFoot =  bool(inputs["Calibration"]["Left flat foot"])
        flag_rightFlatFoot =  bool(inputs["Calibration"]["Right flat foot"])
        markerDiameter = float(inputs["Calibration"]["Marker diameter"])
        pointSuffix = inputs["Calibration"]["Point suffix"]


        # subject mp
        subjects = pyNEXUS.GetSubjectNames()
        subject =   subjects[0]
        logging.info(  "Subject name : " + subject  )
        Parameters = pyNEXUS.GetSubjectParamNames(subject)



        required_mp={
        'Bodymass'   : pyNEXUS.GetSubjectParamDetails( subject, "Bodymass")[0],#71.0,
        'LeftLegLength' : pyNEXUS.GetSubjectParamDetails( subject, "LeftLegLength")[0],#860.0,
        'RightLegLength' : pyNEXUS.GetSubjectParamDetails( subject, "RightLegLength")[0],#865.0 ,
        'LeftKneeWidth' : pyNEXUS.GetSubjectParamDetails( subject, "LeftKneeWidth")[0],#102.0,
        'RightKneeWidth' : pyNEXUS.GetSubjectParamDetails( subject, "RightKneeWidth")[0],#103.4,
        'LeftAnkleWidth' : pyNEXUS.GetSubjectParamDetails( subject, "LeftAnkleWidth")[0],#75.3,
        'RightAnkleWidth' : pyNEXUS.GetSubjectParamDetails( subject, "RightAnkleWidth")[0],#72.9,
        }

        optional_mp={
        'InterAsisDistance'   : pyNEXUS.GetSubjectParamDetails( subject, "InterAsisDistance")[0],#0,
        'LeftAsisTrocanterDistance' : pyNEXUS.GetSubjectParamDetails( subject, "LeftAsisTrocanterDistance")[0],#0,
        'LeftTibialTorsion' : pyNEXUS.GetSubjectParamDetails( subject, "LeftTibialTorsion")[0],#0 ,
        'LeftThighRotation' : pyNEXUS.GetSubjectParamDetails( subject, "LeftThighRotation")[0],#0,
        'LeftShankRotation' : pyNEXUS.GetSubjectParamDetails( subject, "LeftShankRotation")[0],#0,
        'RightAsisTrocanterDistance' : pyNEXUS.GetSubjectParamDetails( subject, "RightAsisTrocanterDistance")[0],#0,
        'RightTibialTorsion' : pyNEXUS.GetSubjectParamDetails( subject, "RightTibialTorsion")[0],#0 ,
        'RightThighRotation' : pyNEXUS.GetSubjectParamDetails( subject, "RightThighRotation")[0],#0,
        'RightShankRotation' : pyNEXUS.GetSubjectParamDetails( subject, "RightShankRotation")[0],#0,
        }


        # -----------CGM STATIC CALIBRATION--------------------
        model=cgm.CGM1LowerLimbs()
        model.configure()
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)

        # reader
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))

        # relabel PIG output if processing previously
        cgm.CGM.reLabelPigOutputs(acqStatic)


        # check static marker configuration
        staticMarkerConfiguration= checkCGM1_StaticMarkerConfig(acqStatic)


        # initial static filter
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                            markerDiameter=markerDiameter,
                                            ).compute()

        # decorators
        # mettre dans l ordre KAD- AnkleMed Knee med de maniere a ce que le label duu node a utiliser dans la calibration finale soit updater

        useLeftKJCnodeLabel = "LKJC_chord"
        useLeftAJCnodeLabel = "LAJC_chord"
        useRightKJCnodeLabel = "RKJC_chord"
        useRightAJCnodeLabel = "RAJC_chord"

        if staticMarkerConfiguration["leftKadFlag"]:
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left", displayMarkers = False)
            useLeftKJCnodeLabel = "LKJC_kad"
            useLeftAJCnodeLabel = "LAJC_kad"
        if staticMarkerConfiguration["rightKadFlag"]:
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right", displayMarkers = False)
            useRightKJCnodeLabel = "RKJC_kad"
            useRightAJCnodeLabel = "RAJC_kad"

        if staticMarkerConfiguration["leftMedialAnkleFlag"]:
            modelDecorator.AnkstaticConfigurationleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
            useLeftAJCnodeLabel = "LAJC_mid"
        if staticMarkerConfiguration["rightMedialAnkleFlag"]:
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
            useRightAJCnodeLabel = "RAJC_mid"

        if staticMarkerConfiguration["leftMedialKneeFlag"]:
            modelDecorator.KmodelFiltersneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left")
            useLeftKJCnodeLabel = "LKJC_mid"


        if staticMarkerConfiguration["rightMedialKneeFlag"]:
            modelDecorator.AnkleCalibrationDecorator(model).mpyCGM2.inputsidCondyles(acqStatic, markerDiameter=markerDiameter, side="right")
            useRightKJCnodeLabel = "RKJC_mid"

        if model.decoratedModel:
            # initial static filter
            modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
                               useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
                               markerDiameter=markerDiameter).compute()


        #----update optional_mp inside the vsk ( mean vsk)-----

        th_l = model.getViconThighOffset("Left")
        sh_l = model.getViconShankOffset("Left")
        tt_l = model.getViconTibialTorsion("Left")

        th_r = model.getViconThighOffset("Right")
        sh_r = model.getViconShankOffset("Right")
        tt_r = model.getViconTibialTorsion("Right")

        spf_l,sro_l = model.getViconFootOffset("Left")
        spf_r,sro_r = model.getViconFootOffset("Right")

        abdAdd_l = model.getViconAnkleAbAddOffset("Left")
        abdAdd_r = model.getViconAnkleAbAddOffset("Right")


        pyNEXUS.SetSubjectParam( subject, "InterAsisDistance",round(model.mp_computed["InterAsisDistance"],6))
        pyNEXUS.SetSubjectParam( subject, "LeftAsisTrocanterDistance",round(model.mp_computed["LeftAsisTrocanterDistance"],6))
        pyNEXUS.SetSubjectParam( subject, "LeftThighRotation",round(th_l,6))
        pyNEXUS.SetSubjectParam( subject, "LeftShankRotation",round(sh_l,6))
        pyNEXUS.SetSubjectParam( subject, "LeftTibialTorsion",round(tt_l,6))


        pyNEXUS.SetSubjectParam( subject, "RightAsisTrocanterDistance",round(model.mp_computed["RightAsisTrocanterDistance"],6))
        pyNEXUS.SetSubjectParam( subject, "RightThighRotation",round(th_r,6))
        pyNEXUS.SetSubjectParam( subject, "RightShankRotation",round(sh_r,6))
        pyNEXUS.SetSubjectParam( subject, "RightTibialTorsion",round(tt_r,6))


        pyNEXUS.SetSubjectParam( subject, "LeftStaticPlantFlex",round(spf_l,6))
        pyNEXUS.SetSubjectParam( subject, "LeftStaticRotOff",round(sro_l,6))
        pyNEXUS.SetSubjectParam( subject, "LeftAnkleAbAdd",round(abdAdd_l,6))

        pyNEXUS.SetSubjectParam( subject, "RightStaticPlantFlex",round(spf_r,6))
        pyNEXUS.SetSubjectParam( subject, "RightStaticRotOff",round(sro_r,6))
        pyNEXUS.SetSubjectParam( subject, "RightAnkleAbAdd",round(abdAdd_r,6))



        # -----------CGM RECONSTRUCTION--------------------


        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Native,
                                                  markerDiameter=markerDiameter,
                                                  useForMotionTest=True)

        modMotion.compute()


#        # Joint kinematics
        modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqStatic,"SACR","midASIS","LPSI")
        modelFilters.ModelAbsoluteAnglesFilter(model,acqStatic,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        # add metadata
        md_Model = btk.btkMetaData('MODEL') # create main metadata
        btk.btkMetaDataCreateChild(md_Model, "NAME", "CGM1")
        btk.btkMetaDataCreateChild(md_Model, "PROCESSOR", "pyCGM2")
        acqStatic.GetMetaData().AppendChild(md_Model)

        # save
        #btkTools.smartWriter(acqStatic,str(DATA_PATH + calibrateFilenameLabelled[:-4] + "_cgm1.c3d"))
        #logging.info( "[pyCGM2] : file ( %s) reconstructed in pyCGM2-model path " % (calibrateFilenameLabelled))

        # save pycgm2.model

        modelFile = open(DATA_PATH + "pyCGM2.model", "w")
        cPickle.dump(model, modelFile)
        modelFile.close()


#        # TEST INTERFACE
#        print "------------------"
#        print pyNEXUS.GetModelOutputNames("cgm1")
#        print "------------------"
#
#        angles=[]
#        for it in btk.Iterate(acqStatic.GetPoints()):
#            if it.GetType() == btk.btkPoint.Angle:
#                angles.append(it.GetLabel())
#        print angles

        logging.info( "EXPORT VICON")
        viconInterface.ViconInterface(pyNEXUS,model,acqStatic,"cgm1").do()

        #pyNEXUS.SaveTrial(30)


    else:
        print "NO Nexus"
