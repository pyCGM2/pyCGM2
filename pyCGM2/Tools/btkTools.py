# coding: utf-8
import numpy as np
from scipy import spatial
import logging

try:
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk




# --- acquisition -----
def smartReader(filename,translators=None):
    """
        Convenient function to read a c3d with Btk

        :Parameters:
            - `filename` (str) - path and filename of the c3d
            - `translators` (str) - marker translators
    """
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename)
    reader.Update()
    acq=reader.GetOutput()
    if translators is not None:
        acq =  applyTranslators(acq,translators)

    # management force plate type 5
    if checkForcePlateExist(acq):
        if "5" in smartGetMetadata(acq,"FORCE_PLATFORM","TYPE"):
            logging.warning("[pyCGM2] Type 5 Force plate detected. Due to a BTK known-issue,  type 5 force plate has been corrected as type 2")
            from pyCGM2.ForcePlates import forceplates # inelegant code but avoir circular import !!
            forceplates.correctForcePlateType5(acq)

    # sort events
    sortedEvents(acq)

    return acq

def smartWriter(acq, filename):
    """
        Convenient function to write a c3d with Btk

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `filename` (str) - path and filename of the c3d
    """
    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(filename)
    writer.Update()




def GetMarkerNames(acq):
    """
    """

    markerNames=[]
    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Marker and it.GetLabel()[0] !="*":
            markerNames.append(it.GetLabel())
    return markerNames


def findNearestMarker(acq,i,marker,markerNames=None):
    values = acq.GetPoint(marker).GetValues()[i,:]

    if markerNames is None:
        markerNames=[]
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetType() == btk.btkPoint.Marker and it.GetLabel()[0] !="*" and it.GetLabel() != marker:
                markerNames.append(it.GetLabel())

    j=0
    out = np.zeros((len(markerNames),3))
    for name in markerNames :
        out[j,:] = acq.GetPoint(name).GetValues()[i,:]
        j+=1

    tree = spatial.KDTree(out)
    dist,index = tree.query(values)


    return markerNames[index],dist




def GetAnalogNames(acq):
    analogNames=[]
    for it in btk.Iterate(acq.GetAnalogs()):
        analogNames.append(it.GetLabel())
    return analogNames


def isGap(acq, markerLabel):
    """
        Check if there is a gap

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerList` (list of str) - marker labels
    """
    residualValues = acq.GetPoint(markerLabel).GetResiduals()
    if any(residualValues== -1.0):
        logging.warning("[pyCGM2] gap found for marker (%s)"%(markerLabel))
        return True
    else:
        return False

def findMarkerGap(acq):
    gaps = list()
    markerNames  = GetMarkerNames(acq)
    for marker in markerNames:
        if isGap(acq,marker):
            gaps.append(marker)

    return gaps



def isPointExist(acq,label):
    """
        Check if a point label exists inside an acquisition

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `label` (str) - point label
    """
    if acq.GetPointNumber()==0:
        return False
    else:
        i = acq.GetPoints().Begin()
        while i != acq.GetPoints().End():
            if i.value().GetLabel()== label:
                flagPoint= True
                break
            else:
                i.incr()
                flagPoint= False

        if flagPoint:
            return True
        else:
            return False

def isPointsExist(acq,labels):
    """
        Check if point labels exist inside an acquisition

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `labels` (list of str) - point labels
    """
    for label in labels:
        if not isPointExist(acq,label):
            logging.warning("[pyCGM2] markers (%s) doesn't exist"% label )
            return False
    return True

def smartAppendPoint(acq,label,values, PointType=btk.btkPoint.Marker,desc="",residuals = None):
    """
        Append/Update a point inside an acquisition

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `label` (str) - point label
            - `values` (numpy.array(n,3)) - point label
            - `PointType` (enums of btk.btkPoint) - type of Point
    """

    logging.debug("new point (%s) added to the c3d" % label)

    # TODO : si value = 1 lignes alors il faudrait dupliquer la lignes pour les n franes
    # valueProj *np.ones((aquiStatic.GetPointFrameNumber(),3))

    values = np.nan_to_num(values)

    if residuals is None:
        residuals = np.zeros((values.shape[0]))
        for i in range(0, values.shape[0]):
            if np.all(values[i,:] == np.zeros((3))):
                residuals[i] = -1
            else:
                residuals[i] = 0

    if isPointExist(acq,label):
        acq.GetPoint(label).SetValues(values)
        acq.GetPoint(label).SetDescription(desc)
        acq.GetPoint(label).SetType(PointType)
        acq.GetPoint(label).SetResiduals(residuals)

    else:
        new_btkPoint = btk.btkPoint(label,acq.GetPointFrameNumber())
        new_btkPoint.SetValues(values)
        new_btkPoint.SetDescription(desc)
        new_btkPoint.SetType(PointType)
        new_btkPoint.SetResiduals(residuals)
        acq.AppendPoint(new_btkPoint)

def clearPoints(acq, pointlabelList):
    """
        Clear points

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `lapointlabelListel` (list of str) - point labels

    """

    i = acq.GetPoints().Begin()
    while i != acq.GetPoints().End():
        label =  i.value().GetLabel()
        if label not in pointlabelList:
            i = acq.RemovePoint(i)
            logging.debug( label + " removed")
        else:
            i.incr()
            logging.debug( label + " found")



    return acq

def checkFirstAndLastFrame (acq, markerLabel):
    """
        Check if extremity frames are correct

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerLabel` (str) - marker label
    """

    if acq.GetPoint(markerLabel).GetValues()[0,0] == 0:
        raise Exception ("[pyCGM2] no marker on first frame")

    if acq.GetPoint(markerLabel).GetValues()[-1,0] == 0:
        raise Exception ("[pyCGM2] no marker on last frame")

def isGap_inAcq(acq, markerList):
    """
        Check if there is a gap

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerList` (list of str) - marker labels
    """
    for m in markerList:
         residualValues = acq.GetPoint(m).GetResiduals()
         if any(residualValues== -1.0):
             raise Exception("[pyCGM2] gap founded for markers %s " % m )

def findValidFrames(acq,markerLabels):

    flag = list()
    for i in range(0,acq.GetPointFrameNumber()):
        pointFlag=list()
        for marker in markerLabels:
            if acq.GetPoint(marker).GetResidual(i) >= 0 :
                pointFlag.append(1)
            else:
                pointFlag.append(0)

        if all(pointFlag)==1:
            flag.append(1)
        else:
            flag.append(0)

    firstValidFrame = flag.index(1)
    lastValidFrame = len(flag) - flag[::-1].index(1) - 1

    return flag,firstValidFrame,lastValidFrame


def applyValidFramesOnOutput(acq,validFrames):

    validFrames = np.asarray(validFrames)

    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() in [btk.btkPoint.Angle, btk.btkPoint.Force, btk.btkPoint.Moment,btk.btkPoint.Power]:
            values = it.GetValues()
            for i in range(0,3):
                values[:,i] =  values[:,i] * validFrames
            it.SetValues(values)

def checkMultipleSubject(acq):
    if acq.GetPoint(0).GetLabel().count(":"):
        raise Exception("[pyCGM2] Your input static c3d was saved with two activate subject. Re-save it with only one before pyCGM2 calculation")


# --- Model -----

def applyTranslators(acq, translators):
    """
    Rename marker from translators
    :Parameters:
        - `acq` (btkAcquisition) - a btk acquisition instance
        - `translators` (dict) - translators
    """
    acqClone = btk.btkAcquisition.Clone(acq)

    if translators is not None:
        modifiedMarkerList = list()

        # gather all labels
        for it in translators.items():
            wantedLabel,initialLabel = it[0],it[1]
            if wantedLabel != initialLabel and initialLabel !="None":
                modifiedMarkerList.append(it[0])
                modifiedMarkerList.append(it[1])

        # Remove Modified Markers from Clone
        for point in  btk.Iterate(acq.GetPoints()):
            if point.GetType() == btk.btkPoint.Marker:
                label = point.GetLabel()
                if label in modifiedMarkerList:
                    acqClone.RemovePoint(label)
                    logging.debug("point (%s) remove in the clone acq  " %((label)))

        # Add Modify markers to clone
        for it in translators.items():
            wantedLabel,initialLabel = it[0],it[1]
            if initialLabel !="None":
                if isPointExist(acq,wantedLabel):
                    smartAppendPoint(acqClone,(wantedLabel+"_origin"),acq.GetPoint(wantedLabel).GetValues(),PointType=btk.btkPoint.Marker) # modified marker
                    logging.warning("wantedLabel (%s)_origin created" %((wantedLabel)))
                if isPointExist(acq,initialLabel):
                    if initialLabel in translators.keys():
                        if translators[initialLabel] == "None":
                            logging.warning("Initial point (%s)and (%s) point to similar values" %((initialLabel), (wantedLabel)))
                            smartAppendPoint(acqClone,(wantedLabel),acq.GetPoint(initialLabel).GetValues(),PointType=btk.btkPoint.Marker)
                            smartAppendPoint(acqClone,(initialLabel),acq.GetPoint(initialLabel).GetValues(),PointType=btk.btkPoint.Marker) # keep initial marker
                        elif translators[initialLabel] == wantedLabel:
                            logging.warning("Initial point (%s) swaped with (%s)" %((initialLabel), (wantedLabel)))
                            initialValue = acq.GetPoint(initialLabel).GetValues()
                            wantedlValue = acq.GetPoint(wantedLabel).GetValues()
                            smartAppendPoint(acqClone,(wantedLabel),initialValue,PointType=btk.btkPoint.Marker)
                            smartAppendPoint(acqClone,("TMP"),wantedlValue,PointType=btk.btkPoint.Marker)
                            acqClone.GetPoint("TMP").SetLabel(initialLabel)
                            acqClone.RemovePoint(wantedLabel+"_origin")
                            acqClone.RemovePoint(initialLabel+"_origin")
                    else:
                        logging.warning("Initial point (%s) renamed (%s)  added into the c3d" %((initialLabel), (wantedLabel)))
                        smartAppendPoint(acqClone,(wantedLabel),acq.GetPoint(initialLabel).GetValues(),PointType=btk.btkPoint.Marker)
                        smartAppendPoint(acqClone,(initialLabel),acq.GetPoint(initialLabel).GetValues(),PointType=btk.btkPoint.Marker)

                else:
                    logging.warning("initialLabel (%s) doesn t exist  " %((initialLabel)))
                    #raise Exception ("your translators are badly configured")


    return acqClone




def checkMarkers( acq, markerList):
    """
        Check if marker labels exist inside an acquisition

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerList` (list of str) - marker labels
    """
    for m in markerList:
        if not isPointExist(acq, m):
            raise Exception("[pyCGM2] markers %s not found" % m )



# --- events -----


def clearEvents(acq,labels):

    events= acq.GetEvents()
    newEvents=btk.btkEventCollection()
    for ev in btk.Iterate(events):
        if ev.GetLabel() not in labels:
            newEvents.InsertItem(ev)

    acq.ClearEvents()
    acq.SetEvents(newEvents)



def modifyEventSubject(acq,newSubjectlabel):
    """
        update the subject name of all events

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `newSubjectlabel` (str) - desired subject name
    """

    # events
    nEvents = acq.GetEventNumber()
    if nEvents>=1:
        for i in range(0, nEvents):
            acq.GetEvent(i).SetSubject(newSubjectlabel)
    return acq

def modifySubject(acq,newSubjectlabel):
    """
        update the subject name inside c3d metadata

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `newSubjectlabel` (str) - desired subject name
    """
    acq.GetMetaData().FindChild("SUBJECTS" ).value().FindChild("NAMES").value().GetInfo().SetValue(0,(newSubjectlabel))


def getNumberOfModelOutputs(acq):
    n_angles=0
    n_forces=0
    n_moments=0
    n_powers=0


    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Angle:
            n_angles+=1

        if it.GetType() == btk.btkPoint.Force:
            n_forces+=1

        if it.GetType() == btk.btkPoint.Moment:
            n_moments+=1

        if it.GetType() == btk.btkPoint.Power:
            n_powers+=1

    return  n_angles,n_forces ,n_moments,n_powers

# --- metadata -----
def hasChild(md,mdLabel):
    """
        Check if a label is within metadata

        .. note::

            btk has a HasChildren method. HasChild doesn t exist, you have to use MetadataIterator to loop metadata

        :Parameters:
            - `md` (btkMetadata) - a btk metadata instance
            - `mdLabel` (str) - label of the metadata you want to check
    """

    outMd = None
    for itMd in btk.Iterate(md):
        if itMd.GetLabel() == mdLabel:
            outMd = itMd
            break
    return outMd


def getVisibleMarkersAtFrame(acq,markers,index):
    visibleMarkers=[]
    for marker in markers:
        if acq.GetPoint(marker).GetResidual(index) !=-1:
            visibleMarkers.append(marker)
    return visibleMarkers




def isAnalogExist(acq,label):
    """
        Check if a point label exists inside an acquisition

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `label` (str) - point label
    """
    #TODO : replace by btkIterate
    i = acq.GetAnalogs().Begin()
    while i != acq.GetAnalogs().End():
        if i.value().GetLabel()==label:
            flag= True
            break
        else:
            i.incr()
            flag= False

    if flag:
        return True
    else:
        return False



def smartAppendAnalog(acq,label,values,desc="" ):

    if isAnalogExist(acq,label):
        acq.GetAnalog(label).SetValues(values)
        acq.GetAnalog(label).SetDescription(desc)
        #acq.GetAnalog(label).SetType(PointType)

    else:
        newAnalog=btk.btkAnalog(acq.GetAnalogFrameNumber())
        newAnalog.SetValues(values)
        newAnalog.SetLabel(label)
        acq.AppendAnalog(newAnalog)


def markerUnitConverter(acq,unitOffset):
    for it in btk.Iterate(acq.GetPoints()):
        if it.GetType() == btk.btkPoint.Marker:
            values = it.GetValues()
            it.SetValues(values*unitOffset)

def constructMarker(acq,label,markers,numpyMethod=np.mean,desc=""):
    nFrames = acq.GetPointFrameNumber()

    x=np.zeros((nFrames, len(markers)))
    y=np.zeros((nFrames, len(markers)))
    z=np.zeros((nFrames, len(markers)))

    i=0
    for marker in markers:
        x[:,i] = acq.GetPoint(marker).GetValues()[:,0]
        y[:,i] = acq.GetPoint(marker).GetValues()[:,1]
        z[:,i] = acq.GetPoint(marker).GetValues()[:,2]
        i+=1

    values = np.zeros((nFrames,3))
    values[:,0] = numpyMethod(x, axis=1)
    values[:,1] = numpyMethod(y, axis=1)
    values[:,2] = numpyMethod(z, axis=1)
    smartAppendPoint(acq,label,values,desc=desc)

def constructEmptyMarker(acq,label,desc=""):
    nFrames = acq.GetPointFrameNumber()
    values = np.zeros((nFrames,3))
    smartAppendPoint(acq,label,values,desc=desc,residuals= np.ones((nFrames,1))*-1.0)


def createZeros(acq, markerLabels):
    for label in markerLabels:
        constructEmptyMarker(acq,label,desc="")


def getNumberOfForcePlate(btkAcq):

    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()

    return pfc.GetItemNumber()


def getStartEndEvents(btkAcq,context,startLabel="Start", endLabel="End"):
    events= btkAcq.GetEvents()

    start=[]
    end=[]
    for ev in btk.Iterate(events):
        if ev.GetContext()== context and ev.GetLabel()==startLabel :
                start.append(ev.GetFrame())
        if ev.GetContext()==context and ev.GetLabel()==endLabel :
                end.append(ev.GetFrame())
    if start==[] or end==[]:
        return None,None

    if len(start)>1 or len(end)>1:
        raise("[pyCGM2]: You can t have multiple Start and End events" )
    elif end<start:
        raise("[pyCGM2]: wrong order ( start<end)" )
    else:
        return start[0],end[0]


def _getSectionFromMd(md):
    md_sections=list()
    for i in range(0, md.GetChildNumber()):
        md_sections.append(md.GetChild(i).GetLabel())
    return md_sections


def changeSubjectName(btkAcq,subjectName):


    # change subject name in the metadata
    md = btkAcq.GetMetaData()


    if "SUBJECTS" in _getSectionFromMd(md):
        subjectMd =  btkAcq.GetMetaData().FindChild("SUBJECTS").value()
        if "NAMES" in _getSectionFromMd(subjectMd):
            subjectMd.FindChild("NAMES").value().GetInfo().SetValue(0,subjectName)

        if "USES_PREFIXES"  not in _getSectionFromMd(subjectMd):
            btk.btkMetaDataCreateChild(subjectMd, "USES_PREFIXES", 0)

    if "ANALYSIS" in _getSectionFromMd(md):
        analysisMd =  btkAcq.GetMetaData().FindChild("ANALYSIS").value()
        if "SUBJECTS" in _getSectionFromMd(analysisMd):
            anaSubMdi = analysisMd.FindChild("SUBJECTS").value().GetInfo()
            for i in range(0,anaSubMdi.GetDimension(1) ):
                anaSubMdi.SetValue(i,subjectName)

    events = btkAcq.GetEvents()
    for ev in btk.Iterate(events):
        ev.SetSubject(subjectName)

    # Do not work
    # eventMd =  btkAcq.GetMetaData().FindChild("EVENT").value()
    # eventMdi = eventMd.FindChild("SUBJECTS").value().GetInfo()
    # for i in range(0,eventMdi.GetDimension(1) ):
    #     eventMdi.SetValue(i,"TEST")

    return btkAcq

def smartGetMetadata(btkAcq,firstLevel,secondLevel):
    md = btkAcq.GetMetaData()
    if secondLevel is not None:
        if firstLevel in _getSectionFromMd(md):
            firstMd =  btkAcq.GetMetaData().FindChild(firstLevel).value()
            if secondLevel in _getSectionFromMd(firstMd):
                return firstMd.FindChild(secondLevel).value().GetInfo().ToString()
    else:
        if firstLevel in _getSectionFromMd(md):
            return md.FindChild(firstLevel).value().GetInfo().ToString()

def smartSetMetadata(btkAcq,firstLevel,secondLevel,index,value):
    md = btkAcq.GetMetaData()
    if firstLevel in _getSectionFromMd(md):
        firstMd =  btkAcq.GetMetaData().FindChild(firstLevel).value()
        if secondLevel in _getSectionFromMd(firstMd):
            return firstMd.FindChild(secondLevel).value().GetInfo().SetValue(index,value)

def checkMetadata(btkAcq,firstLevel,secondLevel):
    md = btkAcq.GetMetaData()
    flag =  False
    if secondLevel is not None:
        if firstLevel in _getSectionFromMd(md):
            firstMd =  btkAcq.GetMetaData().FindChild(firstLevel).value()
            if secondLevel in _getSectionFromMd(firstMd):
                flag =  True
    else:
        if firstLevel in _getSectionFromMd(md):
            flag = True

    return flag

def checkForcePlateExist(btkAcq):

    if checkMetadata(btkAcq,"FORCE_PLATFORM","USED"):
        if  smartGetMetadata(btkAcq,"FORCE_PLATFORM","USED")[0] != "0":
            return True
        else:
            return False
    else:
        return False




def NexusGetTrajectory(acq,label):
    values = acq.GetPoint(label).GetValues()
    residuals = acq.GetPoint(label).GetResiduals()[:,0]
    residualsBool = np.asarray(residuals)==0

    return values[:,0],values[:,1],values[:,2],residualsBool


def sortedEvents(acq):
    evs = acq.GetEvents()

    contextLst=[] # recuperation de tous les contextes
    for it in btk.Iterate(evs):
        if it.GetContext() not in contextLst:
            contextLst.append(it.GetContext())


    valueFrame=[] # recuperation de toutes les frames SANS doublons
    for it in  btk.Iterate(evs):
        if it.GetFrame() not in valueFrame:
            valueFrame.append(it.GetFrame())
    valueFrame.sort() # trie

    events =[]
    for frame in valueFrame:
        for it in  btk.Iterate(evs):
            if it.GetFrame()==frame:
                events.append(it)

    newEvents=btk.btkEventCollection()
    for ev in events:
        newEvents.InsertItem(ev)

    acq.ClearEvents()
    acq.SetEvents(newEvents)


#------------------- FROM trials TOOLS------------------------------------------

def buildTrials(dataPath,filenames):

    acqs=[]
    acqFilenames =[]
    for filename in filenames:
        logging.debug( dataPath)
        logging.debug( filename)
        acq = smartReader(dataPath + filename)
        sortedEvents(acq)

        acqs.append(acq)
        acqFilenames.append(filename)

    return acqs, acqFilenames


def isKineticFlag(acq):
    """
        Flag up if correct kinetics available

        :Parameters:
            - `trial` (openma.trial) - an openma trial instance

        :Return:
            - `` (bool) - flag if kinetic available
            - `kineticEvent_times` (lst) - time of maximal Normal reaction Forces for both context
            - `kineticEvent_times_left` (lst) - time of maximal Normal reaction Forces for the Left context
            - `kineticEvent_times_right` (lst) - time of maximal Normal reaction Forces for the Right context
    """


    kineticEvent_frames=[]
    kineticEvent_frames_left=[]
    kineticEvent_frames_right=[]

    events= acq.GetEvents()
    for ev in btk.Iterate(events):
        if ev.GetContext() == "General":
            if ev.GetLabel() in ["Left-FP","Right-FP"]:
                kineticEvent_frames.append(ev.GetFrame())
            if ev.GetLabel() in ["Left-FP"]:
                kineticEvent_frames_left.append(ev.GetFrame())
            if ev.GetLabel() in ["Right-FP"]:
                kineticEvent_frames_right.append(ev.GetFrame())

    if kineticEvent_frames==[]:
        return False,0,0,0
    else:
        return True,kineticEvent_frames,kineticEvent_frames_left,kineticEvent_frames_right

def automaticKineticDetection(dataPath,filenames,acqs=None):
    """
        convenient method for detecting correct kinetic in a filename set

        :Parameters:
            - `dataPath` (str) - folder path
            - `filenames` (list of str) - filename of the different acquisitions
    """
    kineticAcqs=[]
    kineticFilenames=[]

    i=0
    for filename in filenames:
        if filename in kineticFilenames:
            logging.debug("[pyCGM2] : filename %s duplicated in the input list" %(filename))
        else:
            if acqs is None:
                acq = smartReader(dataPath + filename)

            else:
                acq = acqs[i]

            sortedEvents(acq)
            flag_kinetics,times, times_l, times_r = isKineticFlag(acq)

            if flag_kinetics:
                kineticFilenames.append(filename)
                kineticAcqs.append(acq)
    i+=1

    kineticAcqs = None if kineticAcqs ==[] else kineticAcqs
    flag_kinetics = False if kineticAcqs ==[] else True


    return kineticAcqs,kineticFilenames,flag_kinetics

def getForcePlateWrench(btkAcq,fpIndex=None):
    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    grwf = btk.btkGroundReactionWrenchFilter()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    grwf.SetInput(pfc)
    grwc = grwf.GetOutput()
    grwc.Update()

    if fpIndex is not None:
        return grwc.GetItem(fpIndex-1)
    else:
        return grwc


def applyRotation(btkAcq,markers,globalFrameOrientation,forwardProgression):
    """
    """
    if globalFrameOrientation == "XYZ":
	    rot = np.array([[1,0,0],[0,1,0],[0,0,1]])
    elif globalFrameOrientation == "YXZ":
	    rot = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    else:
        raise Exception("[pyCGM2] code cannot work with Z as non-normal axis")

    for marker in markers:
    	values = btkAcq.GetPoint(marker).GetValues()

    	valuesRot = np.zeros((btkAcq.GetPointFrameNumber(),3))
    	for i in range (0, btkAcq.GetPointFrameNumber()):
    		valuesRot[i,:]= np.dot(rot,values[i,:])
    		if not forwardProgression:
    			valuesRot[i,:] = np.dot(np.array([[-1,0,0],[0,1,0],[0,0,1]]),valuesRot[i,:])

    	btkAcq.GetPoint(marker).SetValues(valuesRot)
