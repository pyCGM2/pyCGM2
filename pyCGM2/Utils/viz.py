import numpy as np

from pyCGM2.Tools import btkTools

def displayFrame(acq, frameInstance, labels = ["O","X","Y","Z"], offset= 100):

    R = frameInstance.getRotation()
    t = frameInstance.getTranslation()

    Xo= np.dot(R,[offset,0,0])+t
    Yo= np.dot(R,[0,offset,0])+t
    Zo= np.dot(R,[0,0,offset])+t

    btkTools.smartAppendPoint(acq,labels[0],      t* np.ones((acq.GetPointFrameNumber(),3)),desc="")
    btkTools.smartAppendPoint(acq,labels[1],      Xo* np.ones((acq.GetPointFrameNumber(),3)),desc="")
    btkTools.smartAppendPoint(acq,labels[2],      Yo* np.ones((acq.GetPointFrameNumber(),3)),desc="")
    btkTools.smartAppendPoint(acq,labels[3],      Zo* np.ones((acq.GetPointFrameNumber(),3)),desc="")
 