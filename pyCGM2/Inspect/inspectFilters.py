import logging
import pyCGM2



from pyCGM2.Tools import btkTools


class QualityFilter(object):
    def __init__(self,procedure):

        self.m_procedure = procedure


    def run(self):
        self.m_procedure.check()
