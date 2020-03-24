# coding: utf-8
from __future__ import unicode_literals
import pyCGM2
import numpy as np
from scipy import spatial
import logging
import encodings
from pyCGM2 import btk
from pyCGM2.Utils import utils



# --- acquisition -----
def smartReader(filename,translators=None):
    """
        Convenient function to read a c3d with Btk

        :Parameters:
            - `filename` (str) - path and filename of the c3d
            - `translators` (str) - marker translators
    """
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(utils.str(filename))
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
    writer.SetFilename(utils.str(filename))
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
    values = acq.GetPoint(utils.str(marker)).GetValues()[i,:]

    if markerNames is None:
        markerNames=[]
        for it in btk.Iterate(acq.GetPoints()):
            if it.GetType() == btk.btkPoint.Marker and it.GetLabel()[0] !="*" and it.GetLabel() != utils.str(marker):
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
    residualValues = acq.GetPoint(utils.str(markerLabel)).GetResiduals()
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
            if i.value().GetLabel()== utils.str(label):
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
        acq.GetPoint(utils.str(label)).SetValues(values)
        acq.GetPoint(utils.str(label)).SetDescription(utils.str(desc))
        acq.GetPoint(utils.str(label)).SetType(PointType)
        acq.GetPoint(utils.str(label)).SetResiduals(residuals)

    else:
        new_btkPoint = btk.btkPoint(utils.str(label),acq.GetPointFrameNumber())
        new_btkPoint.SetValues(values)
        new_btkPoint.SetDescription(utils.str(desc))
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

    if acq.GetPoint(utils.str(markerLabel)).GetValues()[0,0] == 0:
        raise Exception ("[pyCGM2] no marker on first frame")

    if acq.GetPoint(utils.str(markerLabel)).GetValues()[-1,0] == 0:
        raise Exception ("[pyCGM2] no marker on last frame")

def isGap_inAcq(acq, markerList):
    """
        Check if there is a gap

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `markerList` (list of str) - marker labels
    """
    for m in markerList:
         residualValues = acq.GetPoint(utils.str(m)).GetResiduals()
         if any(residualValues== -1.0):
             raise Exception("[pyCGM2] gap founded for markers %s " % m )

def findValidFrames(acq,markerLabels):

    flag = list()
    for i in range(0,acq.GetPointFrameNumber()):
        pointFlag=list()
        for marker in markerLabels:
            if acq.GetPoint(utils.str(marker)).GetResidual(i) >= 0 :
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
                    acqClone.RemovePoint(utils.str(label))
                    logging.debug("point (%s) remove in the clone acq  " %((label)))

        # Add Modify markers to clone
        for it in translators.items():
            wantedLabel,initialLabel = it[0],it[1]
            if initialLabel !="None":
                print wantedLabel
                if isPointExist(acq,wantedLabel):
                    smartAppendPoint(acqClone,(wantedLabel+"_origin"),acq.GetPoint(utils.str(wantedLabel)).GetValues(),PointType=btk.btkPoint.Marker) # modified marker
                    logging.warning("wantedLabel (%s)_origin created" %((wantedLabel)))
                if isPointExist(acq,initialLabel):
                    if initialLabel in translators.keys():
                        if translators[initialLabel] == "None":
                            logging.warning("Initial point (%s)and (%s) point to similar values" %((initialLabel), (wantedLabel)))
                            smartAppendPoint(acqClone,(wantedLabel),acq.GetPoint(utils.str(initialLabel)).GetValues(),PointType=btk.btkPoint.Marker)
                            smartAppendPoint(acqClone,(initialLabel),acq.GetPoint(utils.str(initialLabel)).GetValues(),PointType=btk.btkPoint.Marker) # keep initial marker
                        elif translators[initialLabel] == wantedLabel:
                            logging.warning("Initial point (%s) swaped with (%s)" %((initialLabel), (wantedLabel)))
                            initialValue = acq.GetPoint(utils.str(initialLabel)).GetValues()
                            wantedlValue = acq.GetPoint(utils.str(wantedLabel)).GetValues()
                            smartAppendPoint(acqClone,(wantedLabel),initialValue,PointType=btk.btkPoint.Marker)
                            smartAppendPoint(acqClone,("TMP"),wantedlValue,PointType=btk.btkPoint.Marker)
                            acqClone.GetPoint(utils.str("TMP")).SetLabel(initialLabel)
                            acqClone.RemovePoint(utils.str(wantedLabel+"_origin"))
                            acqClone.RemovePoint(utils.str(initialLabel+"_origin"))
                    else:
                        logging.warning("Initial point (%s) renamed (%s)  added into the c3d" %((initialLabel), (wantedLabel)))
                        smartAppendPoint(acqClone,(wantedLabel),acq.GetPoint(utils.str(initialLabel)).GetValues(),PointType=btk.btkPoint.Marker)
                        smartAppendPoint(acqClone,(initialLabel),acq.GetPoint(utils.str(initialLabel)).GetValues(),PointType=btk.btkPoint.Marker)

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
            acq.GetEvent(i).SetSubject(utils.str(newSubjectlabel))
    return acq

def modifySubject(acq,newSubjectlabel):
    """
        update the subject name inside c3d metadata

        :Parameters:
            - `acq` (btkAcquisition) - a btk acquisition inctance
            - `newSubjectlabel` (str) - desired subject name
    """
    acq.GetMetaData().FindChild(utils.str("SUBJECTS") ).value().FindChild(utils.str("NAMES")).value().GetInfo().SetValue(0,(utils.str(newSubjectlabel)))


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
        if itMd.GetLabel() == utils.str(mdLabel):
            outMd = itMd
            break
    return outMd


def getVisibleMarkersAtFrame(acq,markers,index):
    visibleMarkers=[]
    for marker in markers:
        if acq.GetPoint(utils.str(marker)).GetResidual(index) !=-1:
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
        if i.value().GetLabel()==utils.str(label):
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
        acq.GetAnalog(utils.str(label)).SetValues(values)
        acq.GetAnalog(utils.str(label)).SetDescription(utils.str(desc))
        #acq.GetAnalog(label).SetType(PointType)

    else:
        newAnalog=btk.btkAnalog(acq.GetAnalogFrameNumber())
        newAnalog.SetValues(values)
        newAnalog.SetLabel(utils.str(label))
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
        x[:,i] = acq.GetPoint(utils.str(marker)).GetValues()[:,0]
        y[:,i] = acq.GetPoint(utils.str(marker)).GetValues()[:,1]
        z[:,i] = acq.GetPoint(utils.str(marker)).GetValues()[:,2]
        i+=1

    values = np.zeros((nFrames,3))
    values[:,0] = numpyMethod(x, axis=1)
    values[:,1] = numpyMethod(y, axis=1)
    values[:,2] = numpyMethod(z, axis=1)
    smartAppendPoint(acq,label,values,desc=desc)

def constructEmptyMarker(acq,label,desc=""):
    nFrames = acq.GetPointFrameNumber()
    values = np.ones((nFrames,3))
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
        if ev.GetContext()== utils.str(context) and ev.GetLabel()==utils.str(startLabel) :
                start.append(ev.GetFrame())
        if ev.GetContext()==utils.str(context) and ev.GetLabel()==utils.str(endLabel) :
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
        subjectMd =  btkAcq.GetMetaData().FindChild(utils.str("SUBJECTS")).value()
        if "NAMES" in _getSectionFromMd(subjectMd):
            subjectMd.FindChild(utils.str("NAMES")).value().GetInfo().SetValue(0,utils.str(subjectName))

        if "USES_PREFIXES"  not in _getSectionFromMd(subjectMd):
            btk.btkMetaDataCreateChild(subjectMd, utils.str("USES_PREFIXES"), 0)

    if "ANALYSIS" in _getSectionFromMd(md):
        analysisMd =  btkAcq.GetMetaData().FindChild(utils.str("ANALYSIS")).value()
        if "SUBJECTS" in _getSectionFromMd(analysisMd):
            anaSubMdi = analysisMd.FindChild(utils.str("SUBJECTS")).value().GetInfo()
            for i in range(0,anaSubMdi.GetDimension(1) ):
                anaSubMdi.SetValue(i,utils.str(subjectName))

    events = btkAcq.GetEvents()
    for ev in btk.Iterate(events):
        ev.SetSubject(utils.str(subjectName))

    # Do not work
    # eventMd =  btkAcq.GetMetaData().FindChild("EVENT").value()
    # eventMdi = eventMd.FindChild("SUBJECTS").value().GetInfo()
    # for i in range(0,eventMdi.GetDimension(1) ):
    #     eventMdi.SetValue(i,"TEST")

    return btkAcq

def smartGetMetadata(btkAcq,firstLevel,secondLevel):
    md = btkAcq.GetMetaData()
    if secondLevel is not None:
        if utils.str(firstLevel) in _getSectionFromMd(md):
            firstMd =  btkAcq.GetMetaData().FindChild(utils.str(firstLevel)).value()
            if utils.str(secondLevel) in _getSectionFromMd(firstMd):
                return firstMd.FindChild(utils.str(secondLevel)).value().GetInfo().ToString()
    else:
        if utils.str(firstLevel) in _getSectionFromMd(md):
            return md.FindChild(utils.str(firstLevel)).value().GetInfo().ToString()

def smartSetMetadata(btkAcq,firstLevel,secondLevel,index,value):
    md = btkAcq.GetMetaData()
    if utils.str(firstLevel) in _getSectionFromMd(md):
        firstMd =  btkAcq.GetMetaData().FindChild(utils.str(firstLevel)).value()
        if utils.str(secondLevel) in _getSectionFromMd(firstMd):
            return firstMd.FindChild(utils.str(secondLevel)).value().GetInfo().SetValue(index,utils.str(value))

def checkMetadata(btkAcq,firstLevel,secondLevel):
    md = btkAcq.GetMetaData()
    flag =  False
    if secondLevel is not None:
        if utils.str(firstLevel) in _getSectionFromMd(md):
            firstMd =  btkAcq.GetMetaData().FindChild(utils.str(firstLevel)).value()
            if utils.str(secondLevel) in _getSectionFromMd(firstMd):
                flag =  True
    else:
        if utils.str(firstLevel) in _getSectionFromMd(md):
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
    values = acq.GetPoint(utils.str(label)).GetValues()
    residuals = acq.GetPoint(utils.str(label)).GetResiduals()[:,0]
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


    #
    # import ipdb; ipdb.set_trace()
    # events =[]
    # for contextLst_it in contextLst:
    #     for frameSort in valueFrame:
    #         for it in  btk.Iterate(evs):
    #             if it.GetFrame()==frameSort and it.GetContext()==contextLst_it:
    #                 events.append(it)
    return events
