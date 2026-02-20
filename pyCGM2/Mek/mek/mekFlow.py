import sys
import yaml
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools

from pyCGM2.Mek.mek import mekTools
from pyCGM2.flow import settingsHandler
import pandas as pd




def create_flowSettings_attribute(settings_dict, ds, groupname):

    content = yaml.dump(settings_dict, allow_unicode=True)

    group = ds.root().retrieve_group(groupname)
    group.create_attribute("flow-userSettings", content)

def read_flowSettings_attribute(ds, groupname):
    group = ds.root().retrieve_group(groupname)
    settingRaw = group.retrieve_attribute("flow-userSettings").read()

    settings_dict = yaml.safe_load(settingRaw)
    return settings_dict


def build_session_conditions_dataframe(ds):

    session_dfs = []

    for group in ds.root().list_group_children_name():
        if "Session" in group:
            settings = read_flowSettings_attribute(ds, f"{group}/Analysis 1")

            detailsDf = settingsHandler.build_session_conditions_dataframe(settings)

            # Copie défensive pour éviter toute modification implicite
            detailsDf = detailsDf.copy()

            # Ajout explicite de la session
            detailsDf["Session"] = group

            session_dfs.append(detailsDf)

    # Concaténation finale
    if session_dfs:
        out = pd.concat(session_dfs, ignore_index=True)
    else:
        out = None

    return out


