import argparse
import copy
import os
import pyCGM2; LOGGER = pyCGM2.LOGGER


# pyCGM2 libraries

from pyCGM2.Tools import btkTools

from pyCGM2.Model.Models import singleBody

def main(args=None):

    if args  is None:
        parser = argparse.ArgumentParser(description='rigid gap Labeling')
        parser.add_argument('--static', type=str, help='filename of the static',required=False)
        parser.add_argument('--target', type=str, help='marker to reconstruct',required=True)
        parser.add_argument('--trackingMarkers', nargs='*', help='list of tracking markers',required=True)
        parser.add_argument('--begin', type=int, help='initial Frame')
        parser.add_argument('--last', type=int, help='last Frame')
        args = parser.parse_args()

    try:
        from viconnexusapi import ViconNexus
        from pyCGM2.Nexus import nexusFilters
        from pyCGM2.Nexus import nexusUtils
        from pyCGM2.Nexus import nexusTools
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        DATA_PATH, reconstructFilenameLabelledNoExt = nexusTools.getTrialName(NEXUS)

        # enfFiles = eclipse.getEnfTrials(DATA_PATH)

        subject = nexusTools.getActiveSubject(NEXUS)

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acqGait = nacf.build()

        ff = acqGait.GetFirstFrame()-1
        lf = acqGait.GetLastFrame()-1

        # static calibration
        if args.static is None:
            staticFilenames = []
            for filename in  os.listdir(DATA_PATH):
                if " Cal " in filename and filename.endswith(".c3d"):
                    staticFilenames.append(filename)

            if len(staticFilenames) == 1:
                LOGGER.logger.info(f"A single static file ( Cal file : {staticFilenames[0]}) has been detected")
                acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilenames[0]))
            elif len(staticFilenames) > 1:
                text = "Mutiple cal file detected, select your file[0]\n"
                i = 0
                for filename in staticFilenames:
                    text = text + "[%i]  %s \n" % (i, filename)
                    i += 1
                reply = input(text)

                if reply == "":
                    acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilenames[0]))
                else:
                    try:
                        reply = int(reply)
                    except ValueError:
                        LOGGER.logger.info("Wrong input. Select the  correct index")

                    acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilenames[reply]))

            else:
                LOGGER.logger.warning("None cal file detected")
                staticFilename = input("input the name of your static file ( without the extension)")
                acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilename+".c3d"))


        else:
            staticFilenameNoExt = args.static
            acqStatic = btkTools.smartReader(str(DATA_PATH+staticFilenameNoExt+".c3d"))

        targetMarker = args.target
        trackingMarkers = args.trackingMarkers

        if args.begin is None and args.last is None: # reconstrution on full frames
            selectInitialFrame = ff
            selectLastFrame = lf
        elif args.begin is not None and args.last is not None: # reconstrution from both selected begin and end frame
            selectInitialFrame = args.begin-1
            selectLastFrame = args.last-1
        elif args.begin is not None and args.last is None: # reconstrution from  selected begin and last frame
            selectInitialFrame = args.begin-1
            selectLastFrame = lf
        elif args.begin is None and args.last is not None: # reconstrution from  first frame and to selected last frame
            selectInitialFrame = ff
            selectLastFrame = args.last-1

       
        rigidBody = singleBody.SingleBody([trackingMarkers[0],trackingMarkers[1],trackingMarkers[2],trackingMarkers[0],targetMarker],
                                    [trackingMarkers[0],trackingMarkers[1],trackingMarkers[2]])
        rigidBody.calibrate(acqStatic) #,calibFramesOfInterest=[66,66])
        rigidBody.fit(acqGait)

        # exemple to add Node
        # globalpos = acqStatic.GetPoint("rigidZ").GetValues().mean(axis=0)
        # rigidBody.addNode("new",globalpos,positionType="Global")

        valReconstruct = rigidBody.getTrajectory(targetMarker)


        if btkTools.isPointExist(acqGait,targetMarker):
            val0 = acqGait.GetPoint(targetMarker).GetValues()
            val_final = copy.deepcopy(val0)
            val_final[selectInitialFrame-ff:selectLastFrame+1-ff,:] = valReconstruct[selectInitialFrame-ff:selectLastFrame+1-ff,:]
        else:
            val_final = valReconstruct
        
        btkTools.smartAppendPoint(acqGait,targetMarker,val_final)

        # nexus display
        if targetMarker in NEXUS.GetMarkerNames(subject):
            LOGGER.logger.info(f"[pyCGM2] {targetMarker} added as a marker")
            nexusTools.setTrajectoryFromArray(NEXUS,subject,targetMarker,val_final,firstFrame = ff)
        else:
            LOGGER.logger.info(f"[pyCGM2] {targetMarker} added as a modelled marker. Update your vsk with a new marker to add it as a marker")
            nexusTools.appendModelledMarkerFromAcq(NEXUS,subject,targetMarker, acqGait,suffix = "")
    else:
        return parser

if __name__ == "__main__":
    main(args=None)
