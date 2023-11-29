from pyCGM2.Utils import prettyfier
from pyCGM2.Utils import files
from bs4 import BeautifulSoup
import os
import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional,Union,Any

class osimInterface(object):
    """
    Interface for interacting with OpenSim models.

    Args:
            data_path (str): Path to the data directory.
            osimFile (str): Filename of the OpenSim model file.
    
    Attributes:
        xml (opensimXmlInterface): An instance of opensimXmlInterface to interact with OpenSim XML files.
    """
    def __init__(self, data_path:str,osimFile:str):
        self.xml = opensimXmlInterface(data_path+osimFile, None)
    
    def getMuscles(self) -> List[str]:
        """
        Retrieves the list of muscles from the OpenSim model.

        Returns:
            List[str]: A list of muscle names.
        """
        items = self.xml.getSoup().find("ForceSet").find("objects").find_all("Thelen2003Muscle")
        muscles =[]
        for it in items:
            muscles.append(it.attrs["name"])
        return muscles

    def getBodies(self) -> List[str]:
        """
        Retrieves the list of bodies from the OpenSim model.

        Returns:
            List[str]: A list of body names.
        """
        items = self.xml.getSoup().find("BodySet").find("objects").find_all("Body")
        bodies =[]
        for it in items:
            bodies.append(it.attrs["name"])
        return bodies


    def getMuscles_bySide(self, addToName: str = "") -> Dict[str, List[str]]:
        """
        Retrieves muscles categorized by their side (left or right).

        Args:
            addToName (str, optional): Additional string to append to each muscle name.

        Returns:
            Dict[str, List[str]]: A dictionary with keys 'Left' and 'Right' mapping to lists of muscle names.
        """
        muscles = self.getMuscles()
        muscleBySide={"Left":[],"Right":[]}
        for muscle in muscles:
            if "_l" in muscle: muscleBySide["Left"].append(muscle+addToName)
            if "_r" in muscle: muscleBySide["Right"].append(muscle+addToName)
        
        return muscleBySide

    def getCoordinates(self) -> List[str]:
        """
        Retrieves the list of joint coordinates from the OpenSim model.

        Returns:
            List[str]: A list of joint coordinate names.
        """
        items = self.xml.getSoup().find("JointSet").find("objects").find_all("Coordinate")
        jointNames =[]
        for it in items:
            jointNames.append(it.attrs["name"])
        return jointNames



class osimCgmInterface(osimInterface):
    """
    Interface for interacting with OpenSim CGM models.

    Inherits from osimInterface.

    Args:
        modelversion (str): The version of the CGM model.

    """

    def __init__(self, modelversion):
        if modelversion == "CGM2.3":
            super(osimCgmInterface,self).__init__(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\","pycgm2-gait2392_simbody.osim")



class opensimXmlInterface(object):
    """
    Interface for interacting with OpenSim XML files.

    Args:
        templateFullFilename (str): Filename of the template OpenSim XML file.
        outFullFilename (Optional[str]): Filename for the output file. If None, derived from templateFullFilename.

    """

    def __init__(self, templateFullFilename:str, outFullFilename:Optional[str]=None):

        if outFullFilename is None:
            outFullFilename = files.getFilename(templateFullFilename)

        self.m_out = outFullFilename
        self.m_soup = BeautifulSoup(open(templateFullFilename), "xml")

    def getSoup(self) -> BeautifulSoup:
        """
        Retrieves the BeautifulSoup object for XML parsing.

        Returns:
            BeautifulSoup: BeautifulSoup object for the OpenSim XML file.
        """
        return self.m_soup

    def set_one(self, labels: Union[str, List[str]], text: str) -> None:
        """
        Sets the text for a specific XML element or a path of elements.

        Args:
            labels (Union[str, List[str]]): The label or path of labels leading to the XML element.
            text (str): The text to set for the XML element.
        """

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

    def set_many(self, label: str, text: str) -> None:
        """
        Sets the text for all XML elements with a specific label.

        Args:
            label (str): The label of the XML elements.
            text (str): The text to set for each XML element.
        """
        items = self.m_soup.find_all(label)
        for item in items:
            item.string = text

    def set_many_inList(self, listName: str, label: str, text: str) -> None:
        """
        Sets the text for all XML elements within a list with a specific label.

        Args:
            listName (str): The label of the list in the XML.
            label (str): The label of the XML elements within the list.
            text (str): The text to set for each XML element in the list.
        """
        items = self.m_soup.find_all(listName)
        for item in items:
            item.find(label).string = text

    def set_inList_fromAttr(self, listName: str, label: str, attrKey: str, attrValue: str, text: str) -> None:
        """
        Sets the text for XML elements within a list based on an attribute's value.

        Args:
            listName (str): The label of the list in the XML.
            label (str): The label of the XML elements within the list.
            attrKey (str): The attribute key to match.
            attrValue (str): The value of the attribute to match.
            text (str): The text to set for the matching XML elements.
        """
        items = self.m_soup.find_all(listName)
        for item in items:
            if item.attrs[attrKey] == attrValue:
                item.find(label).string = text

    def update(self):
        """
        Updates the XML file with changes made to the BeautifulSoup object.

        Writes the updated XML content to the output file specified in `m_out`.
        """
        ugly = self.m_soup.prettify()
        pretty = prettyfier.prettify_xml(ugly)
        with open(self.m_out, "w") as f:
            f.write(pretty)
            pass