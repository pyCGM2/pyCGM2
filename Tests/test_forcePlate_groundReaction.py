# -*- coding: utf-8 -*-
# pytest -s --disable-pytest-warnings  test_forcePlate_groundReaction.py::Test_groundReactionForcePlateIntegration

import matplotlib.pyplot as plt
import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Lib.Processing import progression

import pyCGM2; LOGGER = pyCGM2.LOGGER

from pyCGM2.Lib.CGM import  cgm1
from pyCGM2.Model import modelFilters
from pyCGM2.Model.Procedures import forcePlateIntegrationProcedures
from pyCGM2.Nexus import vskTools
from pyCGM2.Utils import testingUtils
from pyCGM2 import enums

from pyCGM2.Report import plot
from pyCGM2.Lib import analysis#, plot
from pyCGM2.Lib import plot as hlplot
from pyCGM2.Lib import emg

from pyCGM2.Report import plot as reportPlot
from pyCGM2.Report import plotFilters
from pyCGM2.Report.Viewers import plotViewers
from pyCGM2.Report.Viewers import groundReactionPlotViewers
from pyCGM2.Report.Viewers import  comparisonPlotViewers
from pyCGM2.Report import normativeDatasets
from pyCGM2.Utils import files

def getModel(data_path,progressionAxis):
    
    if progressionAxis == "X":
        staticFilename = "static.c3d"
        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vsk = vskTools.Vsk(data_path + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        # calibration according CGM1
        model,finalAcqStatic,error = cgm1.calibrate(data_path,
            staticFilename,
            None,
            required_mp,
            optional_mp,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            pointSuffix,
            displayCoordinateSystem=True)

    if progressionAxis == "Y":

        staticFilename = "static.c3d"

        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vsk = vskTools.Vsk(data_path + "Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        # calibration according CGM1
        model,finalAcqStatic,error = cgm1.calibrate(data_path,
            staticFilename,
            None,
            required_mp,
            optional_mp,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            pointSuffix,
            displayCoordinateSystem=True)

    return model


class Test_GroundReactionForcePlate():

    def test_GroundReaction(self):

        def plot(acq):
            plt.figure()
            plt.plot(acq.GetPoint("LGroundReactionForce").GetValues()[:,0],'-r')
            plt.plot(acq.GetPoint("LGroundReactionForce_check").GetValues()[:,0],'or')

            plt.figure()
            plt.plot(acq.GetPoint("LGroundReactionForce").GetValues()[:,1],'-r')
            plt.plot(acq.GetPoint("LGroundReactionForce_check").GetValues()[:,1],'or')

            plt.figure()
            plt.plot(acq.GetPoint("LGroundReactionForce").GetValues()[:,2],'-r')
            plt.plot(acq.GetPoint("LGroundReactionForce_check").GetValues()[:,2],'or')

            plt.figure()
            plt.plot(acq.GetPoint("RGroundReactionForce").GetValues()[:,0],'-b')
            plt.plot(acq.GetPoint("RGroundReactionForce_check").GetValues()[:,0],'ob')

            plt.figure()
            plt.plot(acq.GetPoint("RGroundReactionForce").GetValues()[:,1],'-b')
            plt.plot(acq.GetPoint("RGroundReactionForce_check").GetValues()[:,1],'ob')

            plt.figure()
            plt.plot(acq.GetPoint("RGroundReactionForce").GetValues()[:,2],'-b')
            plt.plot(acq.GetPoint("RGroundReactionForce_check").GetValues()[:,2],'ob')

        data_path =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions_Xprogression\\"
        model = getModel(data_path,"X")

        #------- X axis forward
        gaitFilename="gait2.c3d"
        acqGaitXf = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "LRLX"
     
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitXf,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitXf,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitXf,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix="check")

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitXf)

        #  testingUtils.test_point_rms(acqGait,"LGroundReactionForce","LGroundReactionForce_test",0.5,init=420, end = 470)
        testingUtils.test_point_rms(acqGaitXf,"LGroundReactionForce","LGroundReactionForce_check",0.5)
        testingUtils.test_point_rms(acqGaitXf,"RGroundReactionForce","RGroundReactionForce_check",0.5)
        #plot(acqGaitXf)
 


        #------ X axis backward
        gaitFilename="gait1.c3d"
        acqGaitXb = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "RLXX"

        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitXb,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitXb,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitXb,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix="check")

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitXb)

        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitXb,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()

        testingUtils.test_point_rms(acqGaitXb,"LGroundReactionForce","LGroundReactionForce_check",0.5)
        testingUtils.test_point_rms(acqGaitXb,"RGroundReactionForce","RGroundReactionForce_check",0.5)
        #plot(acqGaitXb)


        data_path =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\LowerLimb-medMed_Yprogression\\"
        model = getModel(data_path,"Y")

        # Y axis forward
        gaitFilename="gait1.c3d"
        acqGaitYf = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "RLX"

        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitYf,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitYf,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitYf,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix="check")

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitYf)


        testingUtils.test_point_rms(acqGaitYf,"LGroundReactionForce","LGroundReactionForce_check",0.5)
        testingUtils.test_point_rms(acqGaitYf,"RGroundReactionForce","RGroundReactionForce_check",0.5)    
        #plot(acqGaitYf)

         #-- Y axis backward
        gaitFilename="gait2.c3d"
        acqGaitYb = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "XLR"
     
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitYb,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitYb,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitYb,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix="check")

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitYb)

        testingUtils.test_point_rms(acqGaitYb,"LGroundReactionForce","LGroundReactionForce_check",0.5)
        testingUtils.test_point_rms(acqGaitYb,"RGroundReactionForce","RGroundReactionForce_check",0.5)
        #plot(acqGaitYb)
        plt.show()

    

    def test_AdapaterGRF_Xaxis(self):

        def plot(acqForward,acqBackward):
            # ----- plot
            for i in range (0,3):
                plt.figure()
                plt.plot(acqForward.GetPoint("LStandardizedGroundReactionForce").GetValues()[:,i],"-r")
                plt.plot(acqForward.GetPoint("RStandardizedGroundReactionForce").GetValues()[:,i],"-b")
                plt.plot(acqBackward.GetPoint("LStandardizedGroundReactionForce").GetValues()[:,i],"--r")
                plt.plot(acqBackward.GetPoint("RStandardizedGroundReactionForce").GetValues()[:,i],"--b")





        data_path =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions_Xprogression\\"
        model = getModel(data_path,"X")

        #------- X axis forward
        gaitFilename="gait2.c3d"
        acqGaitXf = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "LRLX"
     
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitXf,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitXf,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitXf,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitXf)


        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitXf,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()



         #------ X axis backward
        gaitFilename="gait1.c3d"
        acqGaitXb = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "RLXX"

        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitXb,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitXb,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitXb,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitXb)

        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitXb,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()



        plot(acqGaitXf,acqGaitXb)
        plt.show()

        
    def test_AdapaterGRF_Yaxis(self):

        def plot(acqForward,acqBackward):
            
            # ----- plot
            for i in range (0,3):
                plt.figure()
                plt.plot(acqForward.GetPoint("LStandardizedGroundReactionForce").GetValues()[:,i],"-r")
                plt.plot(acqForward.GetPoint("RStandardizedGroundReactionForce").GetValues()[:,i],"-b")
                plt.plot(acqBackward.GetPoint("LStandardizedGroundReactionForce").GetValues()[:,i],"--r")
                plt.plot(acqBackward.GetPoint("RStandardizedGroundReactionForce").GetValues()[:,i],"--b")



        data_path =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\LowerLimb-medMed_Yprogression\\"
        model = getModel(data_path,"Y")

        #------- Y axis forward
        gaitFilename="gait1.c3d"
        acqGaitYf = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "RLX"
     
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitYf,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitYf,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitYf,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitYf)


        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitYf,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()


         #------ Y axis backward
        gaitFilename="gait2.c3d"
        acqGaitYb = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "XLR"

        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitYb,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitYb,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitYb,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitYb)

        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitYb,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()

        plot(acqGaitYf,acqGaitYb)
        plt.show()

    def test_plots(self):
        data_path =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\LowerLimb-medMed_Yprogression\\"
        model = getModel(data_path,"Y")

        #------- Y axis forward
        gaitFilename="gait1.c3d"
        acqGaitYf = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "RLX"
     
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitYf,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitYf,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)


        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitYf,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitYf)


        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitYf,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()

        btkTools.smartWriter(acqGaitYf,data_path+"gait1.c3d")

         #------ Y axis backward
        gaitFilename="gait2.c3d"
        acqGaitYb = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "XLR"

        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitYb,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitYb,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitYb,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitYb)

        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitYb,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()

        btkTools.smartWriter(acqGaitYb,data_path+"gait2.c3d")

        # -----analysis------

        analysisInstance = analysis.makeAnalysis(data_path,
                        ["gait1.c3d","gait2.c3d"],
                        type="Gait",
                        emgChannels = None,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=None,
                        )
        
        #   filter 1 
        # procedure - filter
        kv = groundReactionPlotViewers.NormalizedGroundReactionForcePlotViewer(analysisInstance,pointLabelSuffix=None)
        kv.setAutomaticYlimits(False)
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        fig = pf.plot()
        #pf.setHorizontalLines({"Vertical Force":[[9.81,"black"]]})
        plt.show()

        
        hlplot.plot_DescriptiveGRF(data_path, analysisInstance, None,
                                   type="Gait",
                                   pointLabelSuffix=None)
        


    

class Test_GRF_Integration():

    def test_NormalGaitIntegration(self): 
        data_path =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\fullBody-native-noOptions\\"   

        staticFilename = "static.c3d"
        markerDiameter=14
        leftFlatFoot = False
        rightFlatFoot = False
        headStraight = False
        pointSuffix = "test"

        vsk = vskTools.Vsk(data_path + "New Subject.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        # calibration according CGM1
        model,finalAcqStatic,error = cgm1.calibrate(data_path,
            staticFilename,
            None,
            required_mp,
            optional_mp,
            leftFlatFoot,
            rightFlatFoot,
            headStraight,
            markerDiameter,
            pointSuffix,
            displayCoordinateSystem=True)

        gaitFilename="gait2.c3d"
        acqGait1 = btkTools.smartReader(data_path +  gaitFilename)
        # mfpa = "RLXX"
        mfpa = "LRLX"
     
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait1,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGait1,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait1)
        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGait1,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix="check")

        # standardize GRF for clinical interpretation 
        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGait1,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()

        # integrate force reaction
        proc = forcePlateIntegrationProcedures.GaitForcePlateIntegrationProcedure()
        modelFilters.GroundReactionIntegrationFilter(proc, acqGait1,mappedForcePlate,model.mp["Bodymass"],
                                                     globalFrameOrientation=globalFrame, forwardProgression=forwardProgression).compute()

        btkTools.smartWriter(acqGait1,data_path+"gait2_stanGRF.c3d")

        acq =  btkTools.smartReader(data_path+"gait2_stanGRF.c3d")
       

        from pyCGM2.Model.CGM2 import cgm

        cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT["Left"] = cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT["Left"] + ["LTotalGroundReactionForce","LCOMTrajectory_FP","LCOMVelocity_FP", "LCOMAcceleration_FP"]
        cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT["Right"] = cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT["Right"] + ["RTotalGroundReactionForce","RCOMTrajectory_FP","RCOMVelocity_FP", "RCOMAcceleration_FP"]


        analysisInstance = analysis.makeAnalysis(data_path,
                        ["gait2_stanGRF.c3d"],
                        type="Gait",
                        kineticLabelsDict =  cgm.CGM.ANALYSIS_KINETIC_LABELS_DICT,
                        emgChannels = None,
                        pointLabelSuffix=None,
                        subjectInfo=None, experimentalInfo=None,modelInfo=required_mp,
                        )

        # # procedure - filter
        kv = groundReactionPlotViewers.NormalizedGaitGrfIntegrationPlotViewer(analysisInstance,bodymass = required_mp["Bodymass"],pointLabelSuffix=None)
        kv.setAutomaticYlimits(False)
        kv.setConcretePlotFunction(plot.gaitDescriptivePlot)
        # # filter
        pf = plotFilters.PlottingFilter()
        pf.setViewer(kv)
        fig = pf.plot()

        hlplot.plot_DescriptiveGrfIntegration(data_path, analysisInstance, None,required_mp["Bodymass"],show=False)

        # #pf.setHorizontalLines({"Vertical Force":[[9.81,"black"]]})

        #hlplot.plot_GaitMeanGrfIntegration(data_path, analysisInstance, None,required_mp["Bodymass"],show=False)


        # hlplot.plot_DescriptiveGRF(data_path, analysisInstance, None,
        #                            type="Gait",
        #                            pointLabelSuffix=None)



        # pf.setHorizontalLines({"Vertical Force":[[9.81,"black"]]})
        plt.show()


    # def test_integration(self): 
    #     data_path =  pyCGM2.TEST_DATA_PATH + "GaitModels\CGM1\\LowerLimb-medMed_Yprogression\\"

    #     model = getModel(data_path,"Y")

    #     analysisInstance = analysis.makeAnalysis(data_path,
    #                     ["gait1.c3d","gait2.c3d"],
    #                     type="Gait",
    #                     emgChannels = None,
    #                     pointLabelSuffix=None,
    #                     subjectInfo=None, experimentalInfo=None,modelInfo=None,
    #                     )

    #     proc =  grfIntegrationProcedures.gaitGrfIntegrationProcedure()
    #     filter = groundReactionIntegrationFilter.GroundReactionIntegrationFilter(analysisInstance,proc,bodymass=model.mp["Bodymass"])
    #     filter.run()