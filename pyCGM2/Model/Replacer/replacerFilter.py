import btk
import pyCGM2; LOGGER = pyCGM2.LOGGER




class ReplacerFilter(object):
    """
    """
    def __init__(self,procedure,nodeLabel,globalPosition):
        self.m_procedure = procedure
        self.m_nodeLabel = nodeLabel
        self.m_globalPosition = globalPosition
    
        self.m_model = None
        self.m_localPosition = None


    def run(self,acq:btk.btkAcquisition):
         self.m_model = self.m_procedure.compute(acq)
         
         self.m_model.getSegment("Body").getReferential("TF").static.addNode(self.m_nodeLabel,self.m_globalPosition,positionType="Global",desc = "Replacer")
         self.m_localPosition = self.m_model.getSegment("Body").getReferential("TF").static.getNode_byLabel(self.m_nodeLabel).getLocal()


    def getModel(self):
        return self.m_model
        
    def getLocalPosition(self):
        return self.m_localPosition
        
        
         
         







        