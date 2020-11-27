# -*- coding: utf-8 -*-
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.path import Path

import re



try: 
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

from pyCGM2.Tools import  btkTools





def appendForcePlateCornerAsMarker (btkAcq):
    """
        Add a marker at each force plate corners

        :Parameters:
           - `btkAcq` (btkAcquisition) : Btk acquisition instance from a c3d

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




def matchingFootSideOnForceplate (btkAcq, enableRefine=True, forceThreshold=50, left_markerLabelToe ="LTOE", left_markerLabelHeel ="LHEE",
                 right_markerLabelToe ="RTOE", right_markerLabelHeel ="RHEE",  display = False, mfpa=None):
    """
        Convenient function detecting foot in contact with a force plate

        **synopsis**

        This function firsly assign foot side to FP from minimal distance with the application point of reaction force.
        A refinement is done subsequently, it confirm if foot side is valid. A foot is invalided if :

         - FP output no data superior to the set threshold
         - Foot markers are not contain in the polygon defined by force plate corner

        :Parameters:
           - `btkAcq` (btkAcquisition) - Btk acquisition instance from a c3d
           - `left_markerLabelToe` (str) - label of the left toe marker
           - `left_markerLabelHeel` (str) - label of the left heel marker
           - `right_markerLabelToe` (str) - label of the right toe marker
           - `right_markerLabelHeel` (str) - label of the right heel marker
           - `display` (bool) - display n figures ( n depend on force plate number) presenting relative distance between mid foot and the orgin of the force plate
           - `mfpa` (string or dict) - manual force plate assigmenment from another method. Can be a string (XLRA, A stand for automatic) or a dict returing assigned foot to a Force plate ID.

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
            logging.info("[pyCGM2]: Id of Force plate not detected")
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
            logging.debug(" Force plate " + str(i) + " : left foot")
            suffix = suffix +  "L"
        else:
            logging.debug(" Force plate " + str(i) + " : right foot")
            suffix = suffix +  "R"

    logging.debug("Matched Force plate ===> %s", (suffix))

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
                logging.debug("PF #%s not activated. It provides no data superior to threshold"%(str(indexFP)) )
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
                containFlags= list()
                for i in range (0, Rz.shape[0]):
                    if boolLst[i]:
                        if path.contains_point(hee[i,0:2]) and path.contains_point(toe[i,0:2]):
                            containFlags.append(True)
                        else:
                            containFlags.append(False)



                if not all(containFlags) == True:
                    logging.debug("PF #%s not activated. While Rz superior to threshold, foot markers are not contained in force plate geometry  "%(str(indexFP)) )
                    # replace only one character
                    li = list(suffix)
                    li[indexFP]="X"
                    suffix ="".join(li)



            indexFP+=1

        # correction with manual assignement
        if mfpa is not None:
            correctedSuffix=""
            if type(mfpa) == dict:
                logging.warning("[pyCGM2] : automatic force plate assigment corrected with context associated with the device Id  ")
                i=0
                for id in pfIDS:
                    fpa = mfpa[id]
                    if fpa != "A":
                        correctedSuffix = correctedSuffix + fpa
                    else:
                        correctedSuffix = correctedSuffix + suffix[i]
                    i+=1
            else:
                logging.warning("[pyCGM2] : automatic force plate assigment corrected  ")
                if len(mfpa) < len(suffix):
                    raise Exception("[pyCGM2] number of assigned force plate inferior to the number of force plate number. Your assignment should have  %s letters at least" %(str(len(suffix))))
                else:
                    if len(mfpa) > len(suffix):
                        logging.warning("[pyCGM2]: Your manual force plate assignement mentions more force plates than the number of force plates stored in the c3d")
                    for i in range(0, len(suffix)):
                        if mfpa[i] != "A":
                            correctedSuffix = correctedSuffix + mfpa[i]
                        else:
                            correctedSuffix = correctedSuffix + suffix[i]
            return correctedSuffix
        else:
            return suffix




def addForcePlateGeneralEvents (btkAcq,mappedForcePlate ):
    """
        Add General events from force plate assignmenet
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

def correctForcePlateType5(btkAcq):

    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()

    md_force_platform =  btkAcq.GetMetaData().FindChild(str("FORCE_PLATFORM")).value()
    md_force_platform_channels = btkAcq.GetMetaData().FindChild(str("FORCE_PLATFORM")).value().FindChild(str("CHANNEL")).value()

    md_force_platform.RemoveChild("CAL_MATRIX")
    md_force_platform.RemoveChild("MATRIX_STORE")

    md_channels=list()
    for i in btkTools.smartGetMetadata(btkAcq,'FORCE_PLATFORM',"CHANNEL"):
        md_channels.append(i)


    pfds = [pfc.GetItem(0),pfc.GetItem(1)]
    forcePlateNumber = len(pfds)


    channel_number_byFp = list()
    for i in range(0,len(pfds)):
        if pfds[i].GetType() in [1,2,4]:
            channel_number_byFp.append(6)
        if pfds[i].GetType() in [3,5]:
            channel_number_byFp.append(8)

    init=0
    channel_indexes_ofAnalog=list()
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
