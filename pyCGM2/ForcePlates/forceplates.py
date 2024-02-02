"""
The module contains convenient functions for working with force plate.

check out the script : *\Tests\test_forcePlateMatching.py* for examples
"""

from typing import Tuple
from typing import Optional
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt
from matplotlib.path import Path
import scipy as sp

import re

import btk
from pyCGM2.Tools import  btkTools

from typing import List, Tuple, Dict, Optional,Union

def ForcePlateIntegration(ReactionForce:np.ndarray, mass:float, frameInit:int=0,frameEnd:int=None,
                            v0:List =[0,0,0], p0:List= [0,0,0], 
                            analogFrequency:int=1000)-> Tuple[np.ndarray, np.ndarray,np.ndarray]:
    """
    Integration of the reaction force to compute position, velocity, and acceleration.

    This function integrates the ground reaction force to calculate the vertical acceleration, velocity, 
    and position of the center of mass.

    Args:
        ReactionForce (np.ndarray): Ground reaction force (array of shape [frames, 3]).
        mass (float): Body mass in kilograms.
        frameInit (int, optional): Initial frame of the area of interest. Defaults to 0.
        frameEnd (int, optional): Final frame of the area of interest. Defaults to None (end of the array).
        v0 (List, optional): Initial velocity (in 3D). Defaults to [0, 0, 0].
        p0 (List, optional): Initial position (in 3D). Defaults to [0, 0, 0].
        analogFrequency (int, optional): Analog frequency in Hz. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of position, velocity, and acceleration.
    """

    g=9.81


    frameInit = int(frameInit)
    frameEnd =  int(ReactionForce.shape[0]) if frameEnd is None else int(frameEnd)
    acceleration0 = np.zeros((ReactionForce.shape))
    velocity0 = np.zeros((ReactionForce.shape))
    position0 = np.zeros((ReactionForce.shape))

    ReactionForce_cut = ReactionForce[frameInit:frameEnd]
    acceleration = np.zeros((ReactionForce_cut.shape))
    velocity = np.zeros((ReactionForce_cut.shape))
    position = np.zeros((ReactionForce_cut.shape))

    # vertical acceleration of the center of mass
    acceleration[:,0] = (ReactionForce_cut[:,0] )/mass
    acceleration[:,1] = (ReactionForce_cut[:,1] )/mass
    acceleration[:,2] = (ReactionForce_cut[:,2] - mass*g)/mass

    for j in range(0,3):
        velocity[:,j] = sp.integrate.cumulative_trapezoid(acceleration[:,j], dx=1/analogFrequency, initial=0)+v0[j]
        position[:,j] = sp.integrate.cumulative_trapezoid(velocity[:,j], dx=1/analogFrequency, initial=0)+p0[j]



    index = 0
    for i in range(frameInit,frameEnd):
        position0[i,:] = position[index,:]
        velocity0[i,:] = velocity[index,:]
        acceleration0[i,:] = acceleration[index,:]
        index+=1

    return position0, velocity0, acceleration0


def appendForcePlateCornerAsMarker (btkAcq:btk.btkAcquisition):
    """
    Update a BTK acquisition by adding force plate corners as markers.

    Args:
        btkAcq (btk.btkAcquisition): A BTK acquisition instance to update.
    """

    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()


    for i in range(0,pfc.GetItemNumber()):
        val_corner0 = pfc.GetItem(i).GetCorner(0).T * np.ones((btkAcq.GetPointFrameNumber(),3))
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner0",val_corner0, desc="forcePlate")

        val_corner1 = pfc.GetItem(i).GetCorner(1).T * np.ones((btkAcq.GetPointFrameNumber(),3))
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner1",val_corner1, desc="forcePlate")

        val_corner2 = pfc.GetItem(i).GetCorner(2).T * np.ones((btkAcq.GetPointFrameNumber(),3))
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner2", val_corner2, desc="forcePlate")

        val_corner3 = pfc.GetItem(i).GetCorner(3).T * np.ones((btkAcq.GetPointFrameNumber(),3))
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "corner3",val_corner3, desc="forcePlate")



        val_origin = (pfc.GetItem(i).GetCorner(0)+
                      pfc.GetItem(i).GetCorner(1)+
                      pfc.GetItem(i).GetCorner(2)+
                      pfc.GetItem(i).GetCorner(3)) /4.0
        val_origin2 = val_origin.T  * np.ones((btkAcq.GetPointFrameNumber(),3))
        btkTools.smartAppendPoint(btkAcq,"fp" + str(i) + "origin",val_origin2, desc="forcePlate")




def matchingFootSideOnForceplate (btkAcq:btk.btkAcquisition, 
                                  enableRefine:bool=True, forceThreshold:int=50, 
                                  left_markerLabelToe:str ="LTOE", left_markerLabelHeel:str ="LHEE",
                                  right_markerLabelToe:str ="RTOE", right_markerLabelHeel:str ="RHEE",  
                                  display:bool = False, mfpa:str=None):
    """
    Detect the foot in contact with a force plate and refine the detection.

    Args:
        btkAcq (btk.btkAcquisition): A BTK acquisition instance.
        enableRefine (bool, optional): Enable refinement based on vertical force and foot markers. Defaults to True.
        forceThreshold (int, optional): Vertical force threshold for force plate detection. Defaults to 50 N.
        left_markerLabelToe (str, optional): Label of the left toe marker. Defaults to "LTOE".
        left_markerLabelHeel (str, optional): Label of the left heel marker. Defaults to "LHEE".
        right_markerLabelToe (str, optional): Label of the right toe marker. Defaults to "RTOE".
        right_markerLabelHeel (str, optional): Label of the right heel marker. Defaults to "RHEE".
        display (bool, optional): Display figures for force plate matching. Defaults to False.
        mfpa (str, optional): Manually assigned force plates. Defaults to None.

    Returns:
        str: Letters indicating the foot assigned to each force plate (e.g., "LRX").
    """


    appendForcePlateCornerAsMarker(btkAcq)

    ff=btkAcq.GetFirstFrame()
    lf=btkAcq.GetLastFrame()
    appf=btkAcq.GetNumberAnalogSamplePerFrame()


    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    grwf = btk.btkGroundReactionWrenchFilter()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    grwf.SetInput(pfc)
    grwc = grwf.GetOutput()
    grwc.Update()

    midfoot_L=(btkAcq.GetPoint(left_markerLabelToe).GetValues() + btkAcq.GetPoint(left_markerLabelHeel).GetValues())/2.0
    midfoot_R=(btkAcq.GetPoint(right_markerLabelToe).GetValues() + btkAcq.GetPoint(right_markerLabelHeel).GetValues())/2.0

    suffix=str()

    if mfpa is not None:
        try:
            pfIDS=[]
            for i in range(0,pfc.GetItemNumber()):
                pfIDS.append( re.findall( "\[(.*?)\]" ,pfc.GetItem(i).GetChannel(0).GetDescription())[0])
        except Exception:
            LOGGER.logger.debug("[pyCGM2]: Id of Force plate not detected")
            pass

    for i in range(0,grwc.GetItemNumber()):
        pos= grwc.GetItem(i).GetPosition().GetValues()
        pos_downsample = pos[0:(lf-ff+1)*appf:appf]   # downsample

        diffL = np.linalg.norm( midfoot_L-pos_downsample,axis =1)
        diffR = np.linalg.norm( midfoot_R-pos_downsample,axis =1)

        if display:
            plt.figure()
            ax = plt.subplot(1,1,1)
            plt.title("Force plate " + str(i+1))
            ax.plot(diffL,'-r')
            ax.plot(diffR,'-b')

        if np.nanmin(diffL)<np.nanmin(diffR):
            LOGGER.logger.debug(" Force plate " + str(i) + " : left foot")
            suffix = suffix +  "L"
        else:
            LOGGER.logger.debug(" Force plate " + str(i) + " : right foot")
            suffix = suffix +  "R"

    LOGGER.logger.debug("Matched Force plate ===> %s", (suffix))

    if enableRefine:
        # refinement of suffix
        indexFP =0

        for letter in suffix:

            force= grwc.GetItem(indexFP).GetForce().GetValues()
            force_downsample = force[0:(lf-ff+1)*appf:appf]   # downsample


            Rz = np.abs(force_downsample[:,2])

            boolLst = Rz > forceThreshold



            enableDataFlag = False
            for it in boolLst.tolist():
                if it == True:
                    enableDataFlag=True

                    break


            if not enableDataFlag:
                LOGGER.logger.debug("PF #%s not activated. It provides no data superior to threshold"%(str(indexFP)) )
                li = list(suffix)
                li[indexFP]="X"
                suffix ="".join(li)

            else:

                if letter =="L":
                    hee = btkAcq.GetPoint(left_markerLabelHeel).GetValues()
                    toe = btkAcq.GetPoint(left_markerLabelToe).GetValues()
                elif letter == "R":
                    hee = btkAcq.GetPoint(right_markerLabelHeel).GetValues()
                    toe = btkAcq.GetPoint(right_markerLabelToe).GetValues()

                # polygon builder
                corner0 =  btkAcq.GetPoint("fp"+ str(indexFP)+"corner0").GetValues()[0,:]
                corner1 =  btkAcq.GetPoint("fp"+ str(indexFP)+"corner1").GetValues()[0,:]
                corner2 =  btkAcq.GetPoint("fp"+ str(indexFP)+"corner2").GetValues()[0,:]
                corner3 =  btkAcq.GetPoint("fp"+ str(indexFP)+"corner3").GetValues()[0,:]

                verts = [
                    corner0[0:2], # left, bottom
                    corner1[0:2], # left, top
                    corner2[0:2], # right, top
                    corner3[0:2], # right, bottom
                    corner0[0:2], # ignored
                    ]

                codes = [Path.MOVETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.CLOSEPOLY,
                         ]

                path = Path(verts, codes)



                # check if contain both toe and hee marker
                containFlags= []
                for i in range (0, Rz.shape[0]):
                    if boolLst[i]:
                        if path.contains_point(hee[i,0:2]) and path.contains_point(toe[i,0:2]):
                            containFlags.append(True)
                        else:
                            containFlags.append(False)



                if not all(containFlags) == True:
                    LOGGER.logger.debug("PF #%s not activated. While Rz superior to threshold, foot markers are not contained in force plate geometry  "%(str(indexFP)) )
                    # replace only one character
                    li = list(suffix)
                    li[indexFP]="X"
                    suffix ="".join(li)



            indexFP+=1

        # correction with manual assignement
        if mfpa is not None:
            correctedSuffix=""
            if type(mfpa) == dict:
                LOGGER.logger.debug("[pyCGM2] : automatic force plate assigment corrected with context associated with the device Id  ")
                i=0
                for id in pfIDS:
                    fpa = mfpa[id]
                    if fpa != "A":
                        correctedSuffix = correctedSuffix + fpa
                    else:
                        correctedSuffix = correctedSuffix + suffix[i]
                    i+=1
            else:
                LOGGER.logger.debug("[pyCGM2] : automatic force plate assigment corrected  ")
                if len(mfpa) < len(suffix):
                    raise Exception("[pyCGM2] number of assigned force plate inferior to the number of force plate number. Your assignment should have  %s letters at least" %(str(len(suffix))))
                else:
                    if len(mfpa) > len(suffix):
                        LOGGER.logger.debug("[pyCGM2]: Your manual force plate assignement mentions more force plates than the number of force plates stored in the c3d")
                    for i in range(0, len(suffix)):
                        if mfpa[i] != "A":
                            correctedSuffix = correctedSuffix + mfpa[i]
                        else:
                            correctedSuffix = correctedSuffix + suffix[i]
            return correctedSuffix
        else:
            return suffix




def addForcePlateGeneralEvents (btkAcq:btk.btkAcquisition,mappedForcePlate:str):
    """
    Add maximum force of force plates as general events in the acquisition.

    Args:
        btkAcq (btk.btkAcquisition): A BTK acquisition instance.
        mappedForcePlate (str): Letters indicating foot side assigned to each force plate (e.g., "LRX").
    """



    ff=btkAcq.GetFirstFrame()
    lf=btkAcq.GetLastFrame()
    pf = btkAcq.GetPointFrequency()
    appf=btkAcq.GetNumberAnalogSamplePerFrame()

     # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    grwf = btk.btkGroundReactionWrenchFilter()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    grwf.SetInput(pfc)
    grwc = grwf.GetOutput()
    grwc.Update()

    # remove force plates events
    btkTools.clearEvents(btkAcq,["Left-FP","Right-FP"])


    # add general events
    indexFP =0
    for letter in mappedForcePlate:

        force= grwc.GetItem(indexFP).GetForce().GetValues()
        force_downsample = force[0:(lf-ff+1)*appf:appf]   # downsample


        Rz = np.abs(force_downsample[:,2])

        frameMax=  ff+np.argmax(Rz)

        if letter == "L":
            ev = btk.btkEvent('Left-FP', (frameMax-1)/pf, 'General', btk.btkEvent.Automatic, '', 'event from Force plate assignment')
            btkAcq.AppendEvent(ev)
        elif letter == "R":
            ev = btk.btkEvent('Right-FP', (frameMax-1)/pf, 'General', btk.btkEvent.Automatic, '', 'event from Force plate assignment')
            btkAcq.AppendEvent(ev)


        indexFP+=1

def correctForcePlateType5(btkAcq:btk.btkAcquisition):
    """
    Correct an acquisition that includes force plates of type 5.

    Args:
        btkAcq (btk.btkAcquisition): A BTK acquisition instance with force plate type 5.

    Returns:
        btk.btkAcquisition: The corrected BTK acquisition instance.
    """

    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()

    md_force_platform =  btkAcq.GetMetaData().FindChild(str("FORCE_PLATFORM")).value()
    md_force_platform_channels = btkAcq.GetMetaData().FindChild(str("FORCE_PLATFORM")).value().FindChild(str("CHANNEL")).value()

    md_force_platform.RemoveChild("CAL_MATRIX")
    md_force_platform.RemoveChild("MATRIX_STORE")

    md_channels=[]
    for i in btkTools.smartGetMetadata(btkAcq,'FORCE_PLATFORM',"CHANNEL"):
        md_channels.append(i)


    pfds = [pfc.GetItem(0),pfc.GetItem(1)]
    forcePlateNumber = len(pfds)


    channel_number_byFp = []
    for i in range(0,len(pfds)):
        if pfds[i].GetType() in [1,2,4]:
            channel_number_byFp.append(6)
        if pfds[i].GetType() in [3,5]:
            channel_number_byFp.append(8)

    init=0
    channel_indexes_ofAnalog=[]
    for i in channel_number_byFp:
        channel_indexes_ofAnalog.append(md_channels[init:init+i])
        init=i


    newChannelIndexes=[]
    for i in range(0,len(pfds)):

        if pfds[i].GetType() == 5:

            numAnalogs = btkAcq.GetAnalogNumber()

            analogChannels =  np.zeros((8,btkAcq.GetAnalogFrameNumber()))
            j=0
            for index in channel_indexes_ofAnalog[i]:
                analogChannels[j,:]=btkAcq.GetAnalog(int(index)-1).GetValues().T
                md_channels.remove(index)
                j+=1

            wrench = np.dot(pfds[i].GetCalMatrix().T.reshape(8,6).T,analogChannels) # warning : storage of cal_matrix of type5 force plate is wrong in btk.

            force = wrench[0:3,:].T
            moment = wrench[3:6,:].T

            origin = pfds[i].GetOrigin()
            corners = pfds[i].GetCorners()


            btkTools.smartAppendAnalog(btkAcq,"Force.Fx"+str(i),force[:,0],desc="virtual Force plate" )
            btkTools.smartAppendAnalog(btkAcq,"Force.Fy"+str(i),force[:,1],desc="virtual Force plate" )
            btkTools.smartAppendAnalog(btkAcq,"Force.Fz"+str(i),force[:,2],desc="virtual Force plate" )

            btkTools.smartAppendAnalog(btkAcq,"Moment.Mx"+str(i),moment[:,0],desc="virtual Force plate" )
            btkTools.smartAppendAnalog(btkAcq,"Moment.My"+str(i),moment[:,1],desc="virtual Force plate" )
            btkTools.smartAppendAnalog(btkAcq,"Moment.Mz"+str(i),moment[:,2],desc="virtual Force plate" )

            new_channel_indexes_ofAnalog = [*range(numAnalogs,numAnalogs+6)]

            numAnalogs = btkAcq.GetAnalogNumber()

            btkTools.smartSetMetadata(btkAcq,'FORCE_PLATFORM',"TYPE",i,str(2))

            newChannelIndexes = newChannelIndexes +  new_channel_indexes_ofAnalog


        else:
            newChannelIndexes = newChannelIndexes + channel_indexes_ofAnalog[i]

    # md_newChannelIndexes= map(lambda x: x + 1, newChannelIndexes)
    md_force_platform_channels.SetInfo(btk.btkMetaDataInfo([6,int(forcePlateNumber)], newChannelIndexes))

    btkAcq.GetMetaData().FindChild(str("FORCE_PLATFORM")).value().FindChild(str("ZERO")).value().SetInfo(btk.btkMetaDataInfo(btk.btkDoubleArray(forcePlateNumber, 0)))

    return btkAcq


def combineForcePlate(acq:btk.btkAcquisition,mappedForcePlate:str)-> Tuple[btk.btkWrench, btk.btkWrench]:
    """
    Combine signals from force plates based on foot side assignment.

    This function aggregates the force, moment, and position data from multiple force plates based on the assignment 
    of each force plate to a specific foot. It creates combined wrenches (forces and moments) for each foot.

    Args:
        acq (btk.btkAcquisition): A BTK acquisition instance.
        mappedForcePlate (str): Letters indicating foot side assigned to each force plate (e.g., "LRX").

    Returns:
        Tuple[btk.btkWrench, btk.btkWrench]: Two btkWrench objects, one for each foot ('Left' and 'Right'). 
                                             Each wrench object contains the combined force, moment, 
                                             and position data for the respective foot. If no data is present 
                                             for a particular foot, its corresponding wrench object will be None.
    """

    analogFrames = acq.GetAnalogFrameNumber()

    matchFp ={"Left":{"Force":[],"Moment":[],"Position":[]},
          "Right":{"Force":[],"Moment":[],"Position":[]}
     }

    for i in range(0,len(mappedForcePlate)):
        if mappedForcePlate[i] == "L":
            grw = btkTools.getForcePlateWrench(acq, fpIndex=i+1)
            matchFp["Left"]["Force"].append(grw.GetForce().GetValues())
            matchFp["Left"]["Moment"].append(grw.GetMoment().GetValues())
            matchFp["Left"]["Position"].append(grw.GetPosition().GetValues())
        if mappedForcePlate[i] == "R":
            grw = btkTools.getForcePlateWrench(acq, fpIndex=i+1)
            matchFp["Right"]["Force"].append(grw.GetForce().GetValues())
            matchFp["Right"]["Moment"].append(grw.GetMoment().GetValues())
            matchFp["Right"]["Position"].append(grw.GetPosition().GetValues())

    if matchFp["Left"]["Force"] !=[]:
        left_wrench = btk.btkWrench()
        LForceBtkPoint = btk.btkPoint(analogFrames)
        LMomentBtkPoint = btk.btkPoint(analogFrames)
        LPositionBtkPoint = btk.btkPoint(analogFrames)

        force = np.sum(matchFp["Left"]["Force"], axis=0)
        moment = np.sum(matchFp["Left"]["Moment"], axis=0)
        position = np.sum(matchFp["Left"]["Position"], axis=0)

        for i in range(0,force.shape[0]):
            if np.abs(force[i,2]) <10:
                position[i,0]=0
                position[i,1]=0
                position[i,2]=0


        LForceBtkPoint.SetValues(force)
        LMomentBtkPoint.SetValues(moment)
        LPositionBtkPoint.SetValues(position)

        left_wrench.SetForce(LForceBtkPoint)
        left_wrench.SetMoment(LMomentBtkPoint)
        left_wrench.SetPosition(LPositionBtkPoint)
    else:
        left_wrench = None

    if matchFp["Right"]["Force"] !=[]:

        right_wrench = btk.btkWrench()
        RForceBtkPoint = btk.btkPoint(analogFrames)
        RMomentBtkPoint = btk.btkPoint(analogFrames)
        RPositionBtkPoint = btk.btkPoint(analogFrames)

        force = np.sum(matchFp["Right"]["Force"], axis=0)
        moment = np.sum(matchFp["Right"]["Moment"], axis=0)
        position = np.sum(matchFp["Right"]["Position"], axis=0)

        for i in range(0,force.shape[0]):
            if np.abs(force[i,2]) <10:
                position[i,0]=0
                position[i,1]=0
                position[i,2]=0

        RForceBtkPoint.SetValues(force)
        RMomentBtkPoint.SetValues(moment)
        RPositionBtkPoint.SetValues(position)

        right_wrench.SetForce(RForceBtkPoint)
        right_wrench.SetMoment(RMomentBtkPoint)
        right_wrench.SetPosition(RPositionBtkPoint)
    else:
        right_wrench = None

    return left_wrench, right_wrench


def detectGaitConsecutiveForcePlates(acq:btk.btkAcquisition,
                                     mappedForcePlate:str, 
                                     threshold:int = 25)-> Optional[Dict]:
    """
    Detect valid and two consecutive foot contacts on force plates, identifying the leading and trailing limbs.

    This function detects instances where both feet are in contact with force plates simultaneously. It is used to 
    identify the leading and trailing limbs during gait, based on the force produced on each force plate. The function 
    validates contacts by ensuring that the force exceeds a threshold and that foot markers are contained within the 
    force plate's area.

    Args:
        acq (btk.btkAcquisition): A BTK acquisition instance.
        mappedForcePlate (str): A string of letters indicating the foot assigned to each force plate (e.g., 'LRX').
        threshold (int, optional): The force threshold for force plate detection. Defaults to 25 N.

    Returns:
        Optional[Dict]: A dictionary with two keys 'Left' and 'Right', each containing a list of force plate indices 
                        indicating the leading and trailing limb contacts. Returns None if no consecutive contacts are detected.
    """
        
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

    consecutive ={"Left":[],
                  "Right":[]}


    indexes0 = list(range(0,len(mappedForcePlate),1))

    index = 0
    for letter in mappedForcePlate:
        indexes =  indexes0.copy()
        indexes.pop(index)
        if letter == "L":
            contactFrame = np.where(fpwc.GetItem(int(index)).GetForce().GetValues()[:,2] >threshold)[0][0]
            forces=np.zeros(len(mappedForcePlate))
            for otherFPindex in indexes:
                force = fpwc.GetItem(int(otherFPindex)).GetForce().GetValues()[contactFrame,2]
                forces[otherFPindex] = force

            if sum(forces>threshold) >1 :
                raise Exception ("[pyCGM2] - more than 2 Force plates are detected ( check your force plate assignement)  ")
            elif sum(forces>threshold) == 0:
                LOGGER.logger.debug("no simultaneous contact")
            else:
                otherFPindex = np.where(forces>threshold)[0][0]
                if mappedForcePlate[otherFPindex] == "R":
                    LOGGER.logger.debug(f" FP#{index}:Left- opposite contact ( right) on force plate {otherFPindex}")
                    leading = index
                    trailing = otherFPindex
                    consecutive["Left"].append([leading, trailing])
 

        if letter == "R":
            contactFrame = np.where(fpwc.GetItem(int(index)).GetForce().GetValues()[:,2] >threshold)[0][0]
                        
            forces=np.zeros(len(mappedForcePlate))
            for otherFPindex in indexes:
                force = fpwc.GetItem(int(otherFPindex)).GetForce().GetValues()[contactFrame,2]
                forces[otherFPindex] = force

            if sum(forces>threshold) >1 :
                raise Exception ("[pyCGM2] - more than 2 Force plates are detected ( check your force plate assignement)  ")
            elif sum(forces>threshold) == 0:
                LOGGER.logger.debug("no simultaneous contact")
            else:
                otherFPindex = np.where(forces>threshold)[0][0]
                LOGGER.logger.debug(f" FP#{index}:Right- opposite contact ( Left) on force plate {otherFPindex}")
                if mappedForcePlate[otherFPindex] == "L":
                    leading = index
                    trailing = otherFPindex
                    consecutive["Right"].append([leading, trailing])
        index+=1
    if consecutive == {"Left":[], "Right":[]}:
        return None
    else:
        return consecutive

