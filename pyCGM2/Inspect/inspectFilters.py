import logging
import pyCGM2



from pyCGM2.Tools import btkTools


class QualityFilter(object):
    def __init__(self,procedure):
        self.m_procedure = procedure
        self.exceptionMode = False

    def enableExceptionMode(self):
        self.m_procedure.exceptionMode = True
        self.exceptionMode = True

    def disableExceptionMode(self):
        self.m_procedure.exceptionMode = False
        self.exceptionMode = False


    def run(self):
        logging.info("----Quality test: %s ----"%(self.m_procedure.title))
        self.m_procedure.check()
        if self.m_procedure.state:
            logging.info("quality Test  => OK :-)")
        else :
            if not self.exceptionMode:
                logging.info("quality Test => FAILED :-(")
            else:
                raise Exception ("[pyCGM2] Quality test (%s) fails"%(self.m_procedure.title))
        logging.info("--------------------------")
