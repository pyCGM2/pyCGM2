# -*- coding: utf-8 -*-

import btk


class ViconInterface(object):

    def __init__(self,nexusHandle,iModel, iAcq, vskName ):
        self.m_model = iModel
        self.m_acq = iAcq
        self.m_vskName = vskName
        self.nexusHandle = nexusHandle


    def do(self):
        self.m_model.viconExport(self.nexusHandle,self.m_acq, self.m_vskName)
        
        
