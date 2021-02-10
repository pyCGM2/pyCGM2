# -*- coding: utf-8 -*-
import logging

class AnomalyDetectionFilter(object):
    def __init__(self,acq,filename,procedure):
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename

    def run(self):

        out = self.m_procedure.run(self.m_acq,self.m_filename)

        return out


class AnomalyCorrectionFilter(object):
    def __init__(self,acq,filename,procedure):
        self.m_procedure = procedure
        self.m_acq = acq
        self.m_filename = filename

    def run(self):

        out = self.m_procedure.run(self.m_acq,self.m_filename)

        return out
