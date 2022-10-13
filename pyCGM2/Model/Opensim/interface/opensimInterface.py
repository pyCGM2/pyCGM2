# -*- coding: utf-8 -*-
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Utils import prettyfier
from pyCGM2.Tools import  btkTools,opensimTools
from pyCGM2.Utils import files
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

class osimInterface(object):
    def __init__(self, data_path,osimFile):
        self.xml = opensimXmlInterface(data_path+osimFile, None)
    
    def getMuscles(self):
        items = self.xml.getSoup().find("ForceSet").find("objects").find_all("Thelen2003Muscle")
        muscles =[]
        for it in items:
            muscles.append(it.attrs["name"])
        return muscles

    def getMuscles_bySide(self,addToName=""):
        muscles = self.getMuscles()
        muscleBySide={"Left":[],"Right":[]}
        for muscle in muscles:
            if "_l" in muscle: muscleBySide["Left"].append(muscle+addToName)
            if "_r" in muscle: muscleBySide["Right"].append(muscle+addToName)
        
        return muscleBySide


class osimCgmInterface(osimInterface):
    def __init__(self, modelversion):
        if modelversion == "CGM2.3":
            super(osimCgmInterface,self).__init__(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\","pycgm2-gait2354_simbody.osim")



class opensimXmlInterface(object):
    def __init__(self, templateFullFilename, outFullFilename=None):

        if outFullFilename is None:
            outFullFilename = files.getFilename(templateFullFilename)

        self.m_out = outFullFilename
        self.m_soup = BeautifulSoup(open(templateFullFilename), "xml")

    def getSoup(self):
        return self.m_soup

    def set_one(self, labels, text):

        if isinstance(labels, str):
            labels = [labels]

        nitems = len(labels)
        if nitems == 1:
            self.m_soup.find(labels[0]).string = text

        else:
            count = 0
            for label in labels:
                if count == 0:
                    current = self.m_soup.find(label)
                else:
                    current = current.find(label)
                count += 1
            current.string = text

    # def set_one(self, label, text):
    #
    #     self.m_soup.find(label).string = text

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