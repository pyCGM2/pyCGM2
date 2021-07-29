# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from bs4 import BeautifulSoup

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Model.Opensim import opensimInterfaceFilters

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim



class opensimInterfaceXmlProcedure(object):
    def __init__(self,DATA_PATH, xmlSetupTemplate):

        self.m_DATA_PATH = DATA_PATH

        self.m_toolFile = DATA_PATH + "/Tool-setup.xml" #"/IKTool-setup.xml"
        self.xml = opensimInterfaceFilters.opensimXmlInterface(xmlSetupTemplate,self.m_toolFile)

    def run(self):
        tool = opensim.InverseKinematicsTool(self.m_ikToolFile)
        tool.run()
