import logging
import pyCGM2



from pyCGM2.Tools import btkTools


class QualityFilter(object):
    def __init__(self,procedure):
        self.m_procedure = procedure

    def enableExceptionMode(self):
        self.m_procedure.exceptionMode = True

    def disableExceptionMode(self):
        self.m_procedure.exceptionMode = False        


    def run(self):
        self.m_procedure.check()
        if self.m_procedure.state:
            logging.info("quality Test => OK :-)")
        else :
            logging.info("quality Test => FAILED :-(")
