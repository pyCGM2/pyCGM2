# -*- coding: utf-8 -*-

class OpensimInterfaceXmlProcedure(object):
    def __init__(self):
        pass

    def setResultsDirname(self,dirname):
        self.m_resultsDir = dirname    

    def setModelVersion(self, modelVersion):
        self.m_modelVersion = modelVersion.replace(".", "")

    def getXml(self):
        return self.xml