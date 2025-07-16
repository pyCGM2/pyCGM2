import sys
import yaml
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools
from pyCGM2.flow import settingsHandler

from pyCGM2.Mek import mekTools





def create_pycgm2Settings_attribute(settings_dict, ds, groupname):

    content = yaml.dump(settings_dict, allow_unicode=True)

    group = ds.root().retrieve_group(groupname)
    group.create_attribute("flow-userSettings", content)

def read_pycgm2Settings_attribute(ds, groupname):
    group = ds.root().retrieve_group(groupname)
    settingRaw = group.retrieve_attribute("flow-userSettings").read()

    settings_dict = yaml.safe_load(settingRaw)
    return settings_dict





