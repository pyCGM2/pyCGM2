# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:46:40 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import scipy as sp
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model import  modelFilters,modelDecorator, frame
import pyCGM2.enums as pyCGM2Enums



class CGM11_calibrationTest():

    @classmethod
    def basicCGM1_manualOffsets(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()
        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 8.95843387169,
        'LeftShankRotation' : 13.8726086688 ,
        'LeftTibialTorsion' : 5,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : -10.0483956768,
        'RightShankRotation' : 15.3638957089,
        'RightTibialTorsion' : 5
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # TESTS
        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )

#        np.testing.assert_almost_equal(model.mp["InterAsisDistance"],model.mp_computed["InterAsisDistance"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["LeftAsisTrocanterDistance"],model.mp_computed["LeftAsisTrocanterDistance"] , decimal = 3)

        np.testing.assert_almost_equal(model.mp["LeftThighRotation"],-1.0*model.mp_computed["LeftThighRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["LeftShankRotation"],-1.0*model.mp_computed["LeftShankRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["LeftTibialTorsion"],-1.0*model.mp_computed["LeftTibialTorsionOffset"] , decimal = 3)

#        np.testing.assert_almost_equal(model.mp["RightAsisTrocanterDistance"],model.mp_computed["RightAsisTrocanterDistance"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["RightThighRotation"],model.mp_computed["RightThighRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["RightShankRotation"],model.mp_computed["RightShankRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["RightTibialTorsion"],model.mp_computed["RightTibialTorsionOffset"] , decimal = 3)


    @classmethod
    def basicCGM1_manualThighShankRotation(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()
        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 8.95843387169,
        'LeftShankRotation' : 13.8726086688 ,
        'LeftTibialTorsion' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : -10.0483956768,
        'RightShankRotation' : 15.3638957089,
        'RightTibialTorsion' : 0
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # TESTS
        np.testing.assert_equal(model.m_useRightTibialTorsion,False )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,False )

#        np.testing.assert_almost_equal(model.mp["InterAsisDistance"],model.mp_computed["InterAsisDistance"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["LeftAsisTrocanterDistance"],model.mp_computed["LeftAsisTrocanterDistance"] , decimal = 3)

        np.testing.assert_almost_equal(model.mp["LeftThighRotation"],-1.0*model.mp_computed["LeftThighRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["LeftShankRotation"],-1.0*model.mp_computed["LeftShankRotationOffset"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["LeftTibialTorsion"],-1.0*model.mp_computed["LeftTibialTorsionOffset"] , decimal = 3)

#        np.testing.assert_almost_equal(model.mp["RightAsisTrocanterDistance"],model.mp_computed["RightAsisTrocanterDistance"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["RightThighRotation"],model.mp_computed["RightThighRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["RightShankRotation"],model.mp_computed["RightShankRotationOffset"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["RightTibialTorsion"],model.mp_computed["RightTibialTorsionOffset"] , decimal = 3)

    @classmethod
    def basicCGM1_manualTibialTorsion(cls):

        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-basic\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()
        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0 ,
        'LeftTibialTorsion' : -10,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 15
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        #TESTS
        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )


#        np.testing.assert_almost_equal(model.mp["InterAsisDistance"],model.mp_computed["InterAsisDistance"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["LeftAsisTrocanterDistance"],model.mp_computed["LeftAsisTrocanterDistance"] , decimal = 3)

#        np.testing.assert_almost_equal(model.mp["LeftThighRotation"],-1.0*model.mp_computed["LeftThighRotationOffset"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["LeftShankRotation"],-1.0*model.mp_computed["LeftShankRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["LeftTibialTorsion"],-1.0*model.mp_computed["LeftTibialTorsionOffset"] , decimal = 3)

#        np.testing.assert_almost_equal(model.mp["RightAsisTrocanterDistance"],model.mp_computed["RightAsisTrocanterDistance"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["RightThighRotation"],model.mp_computed["RightThighRotationOffset"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["RightShankRotation"],model.mp_computed["RightShankRotationOffset"] , decimal = 3)
        np.testing.assert_almost_equal(model.mp["RightTibialTorsion"],model.mp_computed["RightTibialTorsionOffset"] , decimal = 3)


    @classmethod
    def advancedCGM1_kadMed_manualTibialTorsion(cls):
        """
        - constraints on both tibial Torsion But application of a KAD-med calibration
        => tibial Torsion has to be udpated

        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\KAD-Med\\"
        staticFilename = "MRI-US-01, 2008-08-08, 3DGA 02.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()
        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0 ,
        'LeftTibialTorsion' : -10,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 15
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.Kad(model,acqStatic).compute()
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()


        np.testing.assert_equal(model.m_useRightTibialTorsion,True )
        np.testing.assert_equal(model.m_useLeftTibialTorsion,True )
        np.testing.assert_equal(model.mp["LeftTibialTorsion"],0 ) # cancel by the decorator
        np.testing.assert_equal(model.mp["RightTibialTorsion"],0)


        ltt_vicon = np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("LTibialTorsion").value().GetInfo().ToDouble()[0])
        rtt_vicon =np.rad2deg(acqStatic.GetMetaData().FindChild("PROCESSING").value().FindChild("RTibialTorsion").value().GetInfo().ToDouble()[0])

        np.testing.assert_almost_equal(-1.0*model.mp_computed["LeftTibialTorsionOffset"],ltt_vicon, decimal = 3)
        np.testing.assert_almost_equal(model.mp_computed["RightTibialTorsionOffset"],rtt_vicon, decimal = 3)

#        np.testing.assert_almost_equal(model.mp["InterAsisDistance"],model.mp_computed["InterAsisDistance"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["LeftAsisTrocanterDistance"],model.mp_computed["LeftAsisTrocanterDistance"] , decimal = 3)

#        np.testing.assert_almost_equal(model.mp["LeftThighRotation"],-1.0*model.mp_computed["LeftThighRotationOffset"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["LeftShankRotation"],-1.0*model.mp_computed["LeftShankRotationOffset"] , decimal = 3)


#        np.testing.assert_almost_equal(model.mp["RightAsisTrocanterDistance"],model.mp_computed["RightAsisTrocanterDistance"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["RightThighRotation"],model.mp_computed["RightThighRotationOffset"] , decimal = 3)
#        np.testing.assert_almost_equal(model.mp["RightShankRotation"],model.mp_computed["RightShankRotationOffset"] , decimal = 3)
        #np.testing.assert_almost_equal(model.mp["RightTibialTorsion"],model.mp_computed["RightTibialTorsionOffset"] , decimal = 3)

    @classmethod
    def advancedCGM11_KneeMedKad(cls):
        """


        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\CGM1.1\medial\\"
        staticFilename = "static-all.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()
        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0 ,
        'LeftTibialTorsion' : -10,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 15
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic)
        #modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        btkTools.smartWriter(acqStatic,"advancedCGM11_KneeMedKad.c3d")

    @classmethod
    def advancedCGM11_KneeMedOnly(cls):
        """


        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\CGM1.1\medial\\"
        staticFilename = "static-all.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()
        markerDiameter=14
        mp={
        'Bodymass'   : 71.0,
        'LeftLegLength' : 860.0,
        'RightLegLength' : 865.0 ,
        'LeftKneeWidth' : 102.0,
        'RightKneeWidth' : 103.4,
        'LeftAnkleWidth' : 75.3,
        'RightAnkleWidth' : 72.9,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0 ,
        'LeftTibialTorsion' : -10,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 15
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic)
        #modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        btkTools.smartWriter(acqStatic,"advancedCGM11_KneeMedOnly.c3d")

    @classmethod
    def advancedCGM11_KneeMedKad_TrueEquinus(cls):
        """


        """
        MAIN_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1-TESTS\\kad-med-TrueEquinus\\"
        staticFilename = "static.c3d"

        acqStatic = btkTools.smartReader(str(MAIN_PATH +  staticFilename))

        model=cgm.CGM1LowerLimbs()
        model.configure()
        markerDiameter=14
        mp={
        'Bodymass'   : 36.9,
        'LeftLegLength' : 665.0,
        'RightLegLength' : 655.0 ,
        'LeftKneeWidth' : 102.7,
        'RightKneeWidth' : 100.2,
        'LeftAnkleWidth' : 64.5,
        'RightAnkleWidth' : 63.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        }

        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0 ,
        'LeftTibialTorsion' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0,
        'RightTibialTorsion' : 0
        }

        model.addAnthropoInputParameters(mp,optional=optional_mp)

        # -----------CGM STATIC CALIBRATION--------------------
        scp=modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # cgm decorator

        modelDecorator.Kad(model,acqStatic).compute()
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, side="both")

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        btkTools.smartWriter(acqStatic,"Kad-med-TrueEquinus.c3d")

if __name__ == "__main__":


    logging.info("######## PROCESS CGM 1.1 --- MANUAL ######")
    CGM11_calibrationTest.basicCGM1_manualOffsets() # work
    CGM11_calibrationTest.basicCGM1_manualThighShankRotation() # work
    CGM11_calibrationTest.basicCGM1_manualTibialTorsion() # work
    CGM11_calibrationTest.advancedCGM1_kadMed_manualTibialTorsion() # work
    CGM11_calibrationTest.advancedCGM11_KneeMedKad()
    CGM11_calibrationTest.advancedCGM11_KneeMedOnly()
    CGM11_calibrationTest.advancedCGM11_KneeMedKad_TrueEquinus()
    logging.info("######## PROCESS CGM 1.1 --- MANUAL --> Done ######")
