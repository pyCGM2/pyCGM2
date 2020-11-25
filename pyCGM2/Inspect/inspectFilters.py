# -*- coding: utf-8 -*-
import logging

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
            logging.info("----Quality test: %s ----"%(self.m_procedure.title))
            if self.m_procedure.state:
                logging.info("quality Test  => OK :-)")
            else :
                if not self.exceptionMode:
                    logging.info("quality Test => FAILED :-(")
                else:
                    raise Exception ("[pyCGM2] Quality test (%s) fails"%(self.m_procedure.title))
            logging.info("--------------------------")
