# -*- coding: utf-8 -*-
"""
Obsolete module : work with anomaly and inspector modules instead
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER

class QualityFilter(object):
    def __init__(self,procedure,verbose=True):
        self.m_procedure = procedure
        self.exceptionMode = False
        self.verbose = verbose

    def enableExceptionMode(self):
        self.m_procedure.exceptionMode = True
        self.exceptionMode = True

    def disableExceptionMode(self):
        self.m_procedure.exceptionMode = False
        self.exceptionMode = False

    def getState(self):
        return self.m_procedure.state

    def run(self):

        self.m_procedure.check()

        if self.verbose:
            LOGGER.logger.info("----Quality test: %s ----"%(self.m_procedure.title))
            if self.m_procedure.state:
                LOGGER.logger.info("quality Test  => OK :-)")
            else :
                if not self.exceptionMode:
                    LOGGER.logger.info("quality Test => FAILED :-(")
                else:
                    raise Exception ("[pyCGM2] Quality test (%s) fails"%(self.m_procedure.title))
            LOGGER.logger.info("--------------------------")
