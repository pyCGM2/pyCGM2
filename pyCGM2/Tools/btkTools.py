# -*- coding: utf-8 -*-
import numpy as np
import logging

from pyCGM2 import btk

# --- acquisition -----
def smartReader(filename,translators=None):
    """
        Convenient function to read a c3d with Btk

        :Parameters:
            - `filename` (str) - path and filename of the c3d
            - `translators` (str) - marker translators
    """
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(str(filename))
    reader.Update()
    acq=reader.GetOutput()
    if translators is not None:
        acq =  applyTranslators(acq,translators)
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
    writer.SetFilename(str(filename))
    writer.Update()


def isPointExist(acq,label):
    """
        Check if a point label exists inside an acquisition

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `label` (str) - point label
    """
    #TODO : replace by btkIterate
    i = acq.GetPoints().Begin()
    while i != acq.GetPoints().End():
        if i.value().GetLabel()==label:
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
            logging.debug("[pyCGM2] markers (%s) doesn't exist"% label )
            return False
    return True

def smartAppendPoint(acq,label,values, PointType=btk.btkPoint.Marker,desc=""):
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

    if isPointExist(acq,label):
        acq.GetPoint(label).SetValues(values)
        acq.GetPoint(label).SetDescription(desc)
        acq.GetPoint(label).SetType(PointType)

    else:
        residuals = np.zeros((values.shape[0]))
        for i in range(0, values.shape[0]):
            if np.all(values[i,:] == np.zeros((3))):
                residuals[i] = -1
            else:
                residuals[i] = 0

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

def isGap(acq, markerList):
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

def applyTranslators(acq, translators,keepInitial=False):
    """
    Rename marker from translators
    :Parameters:
        - `acq` (btkAcquisition) - a btk acquisition instance
        - `translators` (dict) - translators
        - `keepInitial` (bool) - flag for avoiding to remove initial markers
    """
    acqClone = btk.btkAcquisition.Clone(acq)

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

    # Add Modify markers to clone
    for it in translators.items():
        wantedLabel,initialLabel = it[0],it[1]
        logging.debug("wantedLabel (%s) initialLabel (%s)  " %(str(wantedLabel), str(initialLabel)))
        if wantedLabel != initialLabel and initialLabel !="None":
            if isPointExist(acq,initialLabel):
                logging.debug("Initial point (%s) renamed (%s)  added into the c3d" %(str(initialLabel), str(wantedLabel)))
                smartAppendPoint(acqClone,str(wantedLabel),acq.GetPoint(str(initialLabel)).GetValues(),PointType=btk.btkPoint.Marker) # modified marker
                if keepInitial:
                    smartAppendPoint(acqClone,str(initialLabel),acq.GetPoint(str(initialLabel)).GetValues(),PointType=btk.btkPoint.Marker) # keep initial marker
            else :
                logging.debug("initialLabel (%s) doesn t exist  " %(str(initialLabel)))

    return acqClone



def findProgression(acq,marker):

    if not isPointExist(acq,marker):
        raise Exception( "[pyCGM2] : marker doesn't exist")

    # find valid frames and get the first one
    flag,vff,vlf = findValidFrames(acq,[marker])

    values = acq.GetPoint(marker).GetValues()[vff:vlf,:]

    MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
    absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

    ind = np.argmax(absMaxValues)
    diff = MaxValues[ind]

    if ind ==0 :
        progressionAxis = "X"
        lateralAxis = "Y"
    else:
        progressionAxis = "Y"
        lateralAxis = "X"

    forwardProgression = True if diff>0 else False

    globalFrame = str(progressionAxis+lateralAxis+"Z")

    logging.debug("Progression axis : %s"%(progressionAxis))
    logging.debug("forwardProgression : %s"%(str(forwardProgression)))
    logging.debug("globalFrame : %s"%(str(globalFrame)))

    return   progressionAxis,forwardProgression,globalFrame


def findProgressionAxisFromPelvicMarkers(acq,markers):

    if not isPointsExist(acq,markers):
        raise Exception( "[pyCGM2] : one of pelvic marker doesn't exist")

    # default
    longitudinalAxis="X"
    globalFrame="XYZ"
    forwardProgression = True

    # find valid frames and get the first one
    flag,vff,vlf = findValidFrames(acq,markers)
    index = vff


    originValues = (acq.GetPoint("LPSI").GetValues()[index,:] + acq.GetPoint("RPSI").GetValues()[index,:])/2.0
    longitudinal_extremityValues = (acq.GetPoint("LASI").GetValues()[index,:] + acq.GetPoint("RASI").GetValues()[index,:])/2.0
    lateral_extremityValues = acq.GetPoint("LPSI").GetValues()[index,:]


    a1=(longitudinal_extremityValues-originValues)
    a1=a1/np.linalg.norm(a1)

    a2=(lateral_extremityValues-originValues)
    a2=a2/np.linalg.norm(a2)

    globalAxes = {"X" : np.array([1,0,0]), "Y" : np.array([0,1,0]), "Z" : np.array([0,0,1])}

    # longitudinal axis
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a1,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    longitudinalAxis =  globalAxes.keys()[maxIndex]
    forwardProgression = True if tmp[maxIndex]>0 else False

    # lateral axis
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a2,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    lateralAxis =  globalAxes.keys()[maxIndex]


    # global frame
    if "X" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"X")
    if "Y" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Y")
    if "Z" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Z")

    logging.debug("Longitudinal axis : %s"%(longitudinalAxis))
    logging.debug("forwardProgression : %s"%(str(forwardProgression)))
    logging.debug("globalFrame : %s"%(str(globalFrame)))

    return   longitudinalAxis,forwardProgression,globalFrame

def findProgressionAxisFromLongAxis(acq,originMarker,frontMarker):

    markers = [originMarker,frontMarker]

    if not isPointsExist(acq,markers):
        raise Exception( "[pyCGM2] : one of the  marker set  doesn't exist")

    # default
    longitudinalAxis="X"
    globalFrame="XYZ"
    forwardProgression = True

    # find valid frames and get the first one
    flag,vff,vlf = findValidFrames(acq,markers)
    index = vff


    origin = acq.GetPoint(originMarker).GetValues()[index,:]
    front = acq.GetPoint(frontMarker).GetValues()[index,:]
    z=np.array([0,0,1])


    a1=(front-origin)
    a1=a1/np.linalg.norm(a1)

    a2=np.cross(a1,z)
    a2=a2/np.linalg.norm(a2)

    globalAxes = {"X" : np.array([1,0,0]), "Y" : np.array([0,1,0]), "Z" : np.array([0,0,1])}

    # longitudinal axis
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a1,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    longitudinalAxis =  globalAxes.keys()[maxIndex]
    forwardProgression = True if tmp[maxIndex]>0 else False

    # lateral axis
    tmp=[]
    for axis in globalAxes.keys():
        res = np.dot(a2,globalAxes[axis])
        tmp.append(res)
    maxIndex = np.argmax(np.abs(tmp))
    lateralAxis =  globalAxes.keys()[maxIndex]


    # global frame
    if "X" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"X")
    if "Y" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Y")
    if "Z" not in str(longitudinalAxis+lateralAxis):
        globalFrame = str(longitudinalAxis+lateralAxis+"Z")

    logging.debug("Longitudinal axis : %s"%(longitudinalAxis))
    logging.debug("forwardProgression : %s"%(str(forwardProgression)))
    logging.debug("globalFrame : %s"%(str(globalFrame)))

    return   longitudinalAxis,forwardProgression,globalFrame

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
    acq.GetMetaData().FindChild("SUBJECTS").value().FindChild("NAMES").value().GetInfo().SetValue(0,str(newSubjectlabel))


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
    smartAppendPoint(acq,label,values,desc=desc)


def createZeros(acq, markerLabels):
    for label in markerLabels:
        constructEmptyMarker(acq,label,desc="")


def getNumberOfForcePlate(btkAcq):

    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()

    return pfc.GetItemNumber()


def getStartEndEvents(btkAcq,context,startLabel="start", endLabel="end"):
    events= btkAcq.GetEvents()

    start=[]
    end=[]
    for ev in btk.Iterate(events):
        if ev.GetContext()==context and ev.GetLabel()==startLabel :
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


def changeSubjectName(btkAcq,subjectName):

    def getSectionFromMd(md):
        md_sections=list()
        for i in range(0, md.GetChildNumber()):
            md_sections.append(md.GetChild(i).GetLabel())
        return md_sections


    # change subject name in the metadata
    md = btkAcq.GetMetaData()


    if "SUBJECTS" in getSectionFromMd(md):
        subjectMd =  btkAcq.GetMetaData().FindChild("SUBJECTS").value()
        if "NAMES" in getSectionFromMd(subjectMd):
            subjectMd.FindChild("NAMES").value().GetInfo().SetValue(0,subjectName)

        if "USES_PREFIXES"  not in getSectionFromMd(subjectMd):
            btk.btkMetaDataCreateChild(subjectMd, "USES_PREFIXES", 0)

    if "ANALYSIS" in getSectionFromMd(md):
        analysisMd =  btkAcq.GetMetaData().FindChild("ANALYSIS").value()
        if "SUBJECTS" in getSectionFromMd(analysisMd):
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
