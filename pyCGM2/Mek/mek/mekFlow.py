# import yaml
import pyCGM2
LOGGER = pyCGM2.LOGGER
import pandas as pd
from pyCGM2.Mek.lib import mekLib
from pyCGM2.flow import settingsHandler 




def build_session_conditions_dataframe(ds):

    session_dfs = []

    for group in ds.root().list_group_children_name():
        if "Session" in group:
            settings = mekLib.readYamlAttribute(ds,f"{group}/Analysis 1","flow-userSettings")

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



