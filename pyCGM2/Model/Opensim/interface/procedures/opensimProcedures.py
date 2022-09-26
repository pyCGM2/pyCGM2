# -*- coding: utf-8 -*-

class OpensimInterfaceXmlProcedure(object):
    def __init__(self):
        self.m_autoXml = True
        

    def setAutoXmlPreparation(self,boolean):
        self.m_autoXml=boolean

    def getXml(self):
        return self.xml