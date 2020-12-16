import logging

class EmgMetadata(object):

    def __init__(self,labels, muscles, side, normalActivityEmgs=None):
        self.m_labels = labels
        self.m_muscles = muscles
        self.m_side = side
        self.m_normalActivityEmgs = normalActivityEmgs

        self.combinedEMG=[]
        for i in range(0,len(self.m_labels)):
            self.combinedEMG.append([self.m_labels[i],self.m_side[i], self.m_muscles[i]])

    def getCombinedEMg(self):
        return self.combinedEMG

    def getChannel(self,muscle,side):

        channel =  None
        for it in self.combinedEMG:
            if it[2] == muscle and it[1] == side:
                channel = it[0]
                break
        if channel is None: logging.info("[pyCGM2] EMG label not find for the %s %s"%(side,muscle))
        return channel
