
# -*- coding: utf-8 -*-
import pyCGM2
import numpy as np
import matplotlib.pyplot as plt
import copy

LOGGER = pyCGM2.LOGGER
try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

from pyCGM2.Model import frame
from pyCGM2.Model import motion

from pyCGM2 import enums
from  pyCGM2.Math import euler
from  pyCGM2.Math import numeric
from pyCGM2.Math import derivation
from pyCGM2.Tools import  btkTools
from pyCGM2.Utils import timer

from pyCGM2.ForcePlates import forceplates


class AbstractForcePlateIntegrationProcedure(object):
    def __init__(self):
        pass


class GaitForcePlateIntegrationProcedure(AbstractForcePlateIntegrationProcedure):
    def __init__(self,):
        super(GaitForcePlateIntegrationProcedure,self).__init__()
        
    def compute(self,acq,mappedForcePlate,bodymass,globalFrameOrientation,forwardProgression):
        pfe = btk.btkForcePlatformsExtractor()
        grwf = btk.btkGroundReactionWrenchFilter()
        pfe.SetInput(acq)
        pfc = pfe.GetOutput()
        grwf.SetInput(pfc)
        grwc = grwf.GetOutput()
        grwc.Update()

        appf = acq.GetNumberAnalogSamplePerFrame()
        freq = acq.GetPointFrequency() 
        pfn = acq.GetPointFrameNumber()
        ff = acq.GetFirstFrame()
        afn = acq.GetAnalogFrameNumber()

        fpwf = btk.btkForcePlatformWrenchFilter() # the wrench of the center of the force platform data, expressed in the global frame
        fpwf.SetInput(pfe.GetOutput())
        fpwc = fpwf.GetOutput()
        fpwc.Update()


        if globalFrameOrientation == "XYZ":
            if forwardProgression:
               Rglobal= np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
            else:
                Rglobal= np.array([[-1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 1]])

        if globalFrameOrientation == "YXZ":
            if forwardProgression:
                Rglobal= np.array([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]])
            else:
                Rglobal= np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])


        totalForceL = np.zeros((afn,3))
        comAccelerationL =  np.zeros((afn,3))
        comVelocityL =  np.zeros((afn,3))
        comTrajectoryL =  np.zeros((afn,3))

        totalForceR = np.zeros((afn,3))
        comAccelerationR =  np.zeros((afn,3))
        comVelocityR =  np.zeros((afn,3))
        comTrajectoryR =  np.zeros((afn,3))

        midASIS = 0.5*(acq.GetPoint("LASI").GetValues()+acq.GetPoint("RASI").GetValues())
        velocity = derivation.firstOrderFiniteDifference(midASIS,freq)

        consecutiveContacts = forceplates.detectGaitConsecutiveForcePlates(acq,mappedForcePlate)

        if consecutiveContacts is not None:

            if consecutiveContacts["Right"] != []:
                for it in consecutiveContacts["Right"]:
                    leadingFPindex = it[0]
                    trailingFPindex = it[1]

                    forceLeadingLimb = fpwc.GetItem(int(leadingFPindex)).GetForce().GetValues()
                    forceTrailingLimb = fpwc.GetItem(int(trailingFPindex)).GetForce().GetValues()
                    total =  forceLeadingLimb +  forceTrailingLimb
                    
                    fsLeading = np.where(forceLeadingLimb[:,2] >25)[0][0]
                    foLeading = np.where(forceLeadingLimb[:,2] >25)[0][-1]

                    fsTrailing = np.where(forceTrailingLimb[:,2] >25)[0][0]
                    foTrailing = np.where(forceTrailingLimb[:,2] >25)[0][-1]


                    # width = acq.GetPoint("RHEE").GetValues()[int(fsLeading/appf),1]-acq.GetPoint("LHEE").GetValues()[int(fsTrailing/appf),1]
                    # length = acq.GetPoint("RHEE").GetValues()[int(fsLeading/appf),0]-acq.GetPoint("LHEE").GetValues()[int(fsTrailing/appf),0]
                    # delta = int((fsLeading-fsTrailing)/appf)*(1/freq)
                    # vlat = (width/1000)/delta
                    # vlong = (length/1000)/delta
                    # vert = midASIS[int(fsLeading/appf),2]-midASIS[int(foTrailing/appf),2]
                    # delta = int((fsLeading-foTrailing)/appf)*(1/freq)
                    # vvert = (vert/1000)/delta

                    pos,vel, acc = forceplates.ForcePlateIntegration(total, bodymass, 
                                                                    frameInit=fsLeading,frameEnd=foLeading,
                                                                    v0 =velocity[int(fsLeading/appf),:]/1000, p0= [0,0,0], 
                                                                    analogFrequency=acq.GetAnalogFrequency())
                    total[:fsLeading+1,:] = 0
                    total[foLeading:,:] = 0 
                    totalForceR = totalForceR+total
                    comAccelerationR =  comAccelerationR + acc
                    comVelocityR =  comVelocityR + vel
                    comTrajectoryR = comTrajectoryR + pos
            
            if consecutiveContacts["Left"] != []:
                for it in consecutiveContacts["Left"]:
                    leadingFPindex = it[0]
                    trailingFPindex = it[1]

                    forceLeadingLimb = fpwc.GetItem(int(leadingFPindex)).GetForce().GetValues()
                    forceTrailingLimb = fpwc.GetItem(int(trailingFPindex)).GetForce().GetValues()
                    total =  forceLeadingLimb +  forceTrailingLimb
                    
                    fsLeading = np.where(forceLeadingLimb[:,2] >25)[0][0]
                    foLeading = np.where(forceLeadingLimb[:,2] >25)[0][-1]
                    fsTrailing = np.where(forceTrailingLimb[:,2] >25)[0][0]
                    foTrailing = np.where(forceTrailingLimb[:,2] >25)[0][-1]

                    pos,vel, acc = forceplates.ForcePlateIntegration(total, bodymass, 
                                                                    frameInit=fsLeading,frameEnd=foLeading,
                                                                    v0 =velocity[int(fsLeading/appf),:]/1000, p0= [0,0,0], 
                                                                    analogFrequency=acq.GetAnalogFrequency())

                    total[:fsLeading+1,:] = 0
                    total[foLeading:,:] = 0 
                    totalForceL = totalForceL+total
                    comAccelerationL =  comAccelerationL + acc
                    comVelocityL =  comVelocityL + vel
                    comTrajectoryL = comTrajectoryL + pos



        # for index in range(1, len(mappedForcePlate)):
            
        #     if mappedForcePlate[index] =="R" and mappedForcePlate[index-1]=="L" :
                
        #         # com = acq.GetPoint("CentreOfMass").GetValues()
                
        #         forceLeadingLimb = fpwc.GetItem(int(index)).GetForce().GetValues()
        #         forceTrailingLimb = fpwc.GetItem(int(index-1)).GetForce().GetValues()
        #         total =  forceLeadingLimb +  forceTrailingLimb
                
        #         fsLeading = np.where(forceLeadingLimb[:,2] >25)[0][0]
        #         foLeading = np.where(forceLeadingLimb[:,2] >25)[0][-1]

        #         fsTrailing = np.where(forceTrailingLimb[:,2] >25)[0][0]
        #         foTrailing = np.where(forceTrailingLimb[:,2] >25)[0][-1]

                
                

                


        #         # width = acq.GetPoint("RHEE").GetValues()[int(fsLeading/appf),1]-acq.GetPoint("LHEE").GetValues()[int(fsTrailing/appf),1]
        #         # length = acq.GetPoint("RHEE").GetValues()[int(fsLeading/appf),0]-acq.GetPoint("LHEE").GetValues()[int(fsTrailing/appf),0]
        #         # delta = int((fsLeading-fsTrailing)/appf)*(1/freq)
        #         # vlat = (width/1000)/delta
        #         # vlong = (length/1000)/delta
        #         # vert = midASIS[int(fsLeading/appf),2]-midASIS[int(foTrailing/appf),2]
        #         # delta = int((fsLeading-foTrailing)/appf)*(1/freq)
        #         # vvert = (vert/1000)/delta

        #         pos,vel, acc = forceplates.ForcePlateIntegration(total, bodymass, 
        #                                                          frameInit=fsLeading,frameEnd=foLeading,
        #                                                          v0 =velocity[int(fsLeading/appf),:]/1000, p0= [0,0,0], 
        #                                                          analogFrequency=acq.GetAnalogFrequency())
        #         total[:fsLeading+1,:] = 0
        #         total[foLeading:,:] = 0 
        #         totalForceR = totalForceR+total
        #         comAccelerationR =  comAccelerationR + acc
        #         comVelocityR =  comVelocityR + vel
        #         comTrajectoryR = comTrajectoryR + pos

                
            
        #     if mappedForcePlate[index] =="L" and mappedForcePlate[index-1]=="R":
        #         forceLeadingLimb = fpwc.GetItem(int(index)).GetForce().GetValues()
        #         forceTrailingLimb = fpwc.GetItem(int(index-1)).GetForce().GetValues()
        #         total =  forceLeadingLimb +  forceTrailingLimb
                
        #         fsLeading = np.where(forceLeadingLimb[:,2] >25)[0][0]
        #         foLeading = np.where(forceLeadingLimb[:,2] >25)[0][-1]
        #         fsTrailing = np.where(forceTrailingLimb[:,2] >25)[0][0]
        #         foTrailing = np.where(forceTrailingLimb[:,2] >25)[0][-1]

        #         pos,vel, acc = forceplates.ForcePlateIntegration(total, bodymass, 
        #                                                          frameInit=fsLeading,frameEnd=foLeading,
        #                                                          v0 =velocity[int(fsLeading/appf),:]/1000, p0= [0,0,0], 
        #                                                          analogFrequency=acq.GetAnalogFrequency())

        #         total[:fsLeading+1,:] = 0
        #         total[foLeading:,:] = 0 
        #         totalForceL = totalForceL+total
        #         comAccelerationL =  comAccelerationL + acc
        #         comVelocityL =  comVelocityL + vel
        #         comTrajectoryL = comTrajectoryL + pos


        for i in range (0, acq.GetAnalogFrameNumber()):
            totalForceL[i,:] = np.dot(Rglobal.T,totalForceL[i,:])
            totalForceR[i,:] = np.dot(Rglobal.T,totalForceR[i,:])
            comAccelerationL[i,:] = np.dot(Rglobal.T,comAccelerationL[i,:])
            comAccelerationR[i,:] = np.dot(Rglobal.T,comAccelerationR[i,:])        
            comVelocityL[i,:] = np.dot(Rglobal.T,comVelocityL[i,:])
            comVelocityR[i,:] = np.dot(Rglobal.T,comVelocityR[i,:])        
            comTrajectoryL[i,:] = np.dot(Rglobal.T,comTrajectoryL[i,:])
            comTrajectoryR[i,:] = np.dot(Rglobal.T,comTrajectoryR[i,:])        

        totalForceR[:,1] = -1.0*totalForceR[:,1]
        comAccelerationR[:,1] = -1.0*comAccelerationR[:,1]  
        comVelocityR[:,1] = -1.0*comVelocityR[:,1]  
        comTrajectoryR[:,1] = -1.0*comTrajectoryR[:,1]  

        
        btkTools.smartAppendPoint(acq,"LCOMTrajectory_FP",
                         comTrajectoryL[::appf],
                         PointType="Scalar", desc="FP integration with a gait scheme")
        btkTools.smartAppendPoint(acq,"RCOMTrajectory_FP",
                         comTrajectoryR[::appf],
                         PointType="Scalar", desc="FP integration with a gait scheme")
        btkTools.smartAppendPoint(acq,"LCOMVelocity_FP",
                         comVelocityL[::appf],
                         PointType="Scalar", desc="FP integration with a gait scheme")
        btkTools.smartAppendPoint(acq,"RCOMVelocity_FP",
                         comVelocityR[::appf],
                         PointType="Scalar", desc="FP integration with a gait scheme")
        btkTools.smartAppendPoint(acq,"LCOMAcceleration_FP",
                         comAccelerationL[::appf],
                         PointType="Scalar", desc="FP integration with a gait scheme")
        btkTools.smartAppendPoint(acq,"RCOMAcceleration_FP",
                         comAccelerationR[::appf],
                         PointType="Scalar", desc="FP integration with a gait scheme")

        btkTools.smartAppendPoint(acq,"RTotalGroundReactionForce",
                         totalForceR[::appf],
                         PointType="Force", desc="from two consecutive contact")
        btkTools.smartAppendPoint(acq,"LTotalGroundReactionForce",
                         totalForceL[::appf],
                         PointType="Force", desc="from two consecutive contact")