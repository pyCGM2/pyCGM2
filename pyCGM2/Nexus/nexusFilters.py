# -*- coding: utf-8 -*-

class NexusModelFilter(object):
    def __init__(self,NEXUS,iModel, iAcq, vskName,pointSuffix, staticProcessing = False ):
        """
            Constructor

            :Parameters:
                - `NEXUS` () - Nexus environment
                - `iModel` (pyCGM2.Model.CGM2.Model) - model instance
                - `vskName` (str) . subject name create in Nexus
                - `staticProcessingFlag` (bool`) : flag indicating only static model ouput will be export

        """
        self.m_model = iModel
        self.m_acq = iAcq
        self.m_vskName = vskName
        self.NEXUS = NEXUS
        self.staticProcessing = staticProcessing
        self.m_pointSuffix = pointSuffix if pointSuffix is None else str("_"+pointSuffix)

    def run(self):
        """
            method calling embedded-model method : viconExport
        """
        self.m_model.viconExport(self.NEXUS,self.m_acq, self.m_vskName,self.m_pointSuffix,self.staticProcessing)
