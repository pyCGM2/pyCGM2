# -*- coding: utf-8 -*-
from pyCGM2.Utils import prettyfier
from pyCGM2.Processing import progressionFrame
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Tools import btkTools
from bs4 import BeautifulSoup
import os
import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim


class opensimXmlInterface(object):
    def __init__(self, templateFullFilename, ouFullFilename):

        self.m_out = ouFullFilename
        self.m_soup = BeautifulSoup(open(templateFullFilename), "xml")

    def getSoup(self):
        return self.m_soup

    def set_one(self, label, text):

        self.m_soup.find(label).string = text

    def set_many(self, label, text):
        items = self.m_soup.find_all(label)
        for item in items:
            item.string = text

    def set_many_inList(self, listName, label, text):
        items = self.m_soup.find_all(listName)
        for item in items:
            item.find(label).string = text

    def set_inList_fromAttr(self, listName, label, attrKey, attrValue, text):
        items = self.m_soup.find_all(listName)
        for item in items:
            if item.attrs[attrKey] == attrValue:
                item.find(label).string = text

    def update(self):
        ugly = self.m_soup.prettify()
        pretty = prettyfier.prettify_xml(ugly)
        with open(self.m_out, "w") as f:
            f.write(pretty)
            pass


class opensimInterfaceScalingFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getOsim(self):
        return self.m_procedure.m_osimModel


class opensimInterfaceInverseKinematicsFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acqMotionFinal
