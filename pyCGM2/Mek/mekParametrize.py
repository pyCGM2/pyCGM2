import sys
import yaml
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools
from pyCGM2.flow import settingsHandler

from pyCGM2.Mek import mekTools


def addPyCGM2SettingsToGroup(settingsPathFile, ds, groupname):
    with open(settingsPathFile, "r", encoding="utf-8") as f:
        content = f.read()
    group = ds.root().retrieve_group(groupname)
    group.create_attribute("userSettings", content)

def readPyCGM2Settings(ds, groupname):
    group = ds.root().retrieve_group(groupname)
    settingRaw = group.retrieve_attribute("userSettings").read()

    settings_dict = yaml.safe_load(settingRaw)
    return settings_dict


class mekAnalysisParametrizeFilter(object):
    """

    """

    def __init__(self,mekStorage,group=None):
        self.m_storage = mekStorage.root()
        self.m_group = group


    def run(self,settings, conditionID):

        group = self.m_storage.retrieve_group(f"{self.m_group}")
        parameterGroup =  group.create_group("Parameters")

        conditionItem = settingsHandler.getConditionItem(settings,conditionID)


        mekTools.write_dict_to_hdf5(parameterGroup, conditionItem)



