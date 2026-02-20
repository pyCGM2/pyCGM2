# coding: utf-8
import os
from pyCGM2.Mek.mek import mekOperations, mekTransform
from pyCGM2.Utils import files
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import pyCGM2 
LOGGER = pyCGM2.LOGGER
# LOGGER.setLevel("info")
# LOGGER.set_file_handler("pyCGM2-Mek.log")

from pyCGM2.Nexus import nexus


import argparse

try:

    from pyCGM2.Mek.mek import mekInit
    from pyCGM2.Mek.mek import mekTransform
except ImportError as e:
    LOGGER.logger.error(f"Error importing Mek modules: {e}. Mek functionalities will not be available.")
    raise e    


def main(args=None):

    if args is None:
        parser = argparse.ArgumentParser(description='Import a trial from Vicon Nexus into a mek storage')
        parser.add_argument('-cgm', '--cgmVersion', type=str,
                            help='userSettings file name, should be in the data folder',
                            required=False, default=None)

        args = parser.parse_args()
    
    cgmVersion = args.cgmVersion


    h5fileOut = f"storage.h5"

    nexusCon = nexus.NexusConnection()
    if nexusCon.isConnected():
        try:
            data_path, trialFilename = nexusCon.nexusTools.getTrialName(nexusCon.NEXUS)
            print(f"trial loaded from Nexus: {trialFilename}")
        except Exception as e:
            LOGGER.logger.warning(f"No trial  loaded in Nexus: {e}")
            raise Exception("No trial  loaded in Nexus")

        h5pathFileOut = data_path + h5fileOut
        h5pathFile = h5pathFileOut if os.path.exists(h5pathFileOut) else None

        storagefilter = mekInit.mekInitStorageFilter(storagePathFile=h5pathFile)
        ds = storagefilter.getStorage()

        proc = mekTransform.mekViconTrialTransformProcedure(cgmVersion=cgmVersion)
        filter = mekTransform.mekTrialTransformFilter(ds,procedure=proc)
        filter.run(data_path,trialFilename+".c3d")

        
        if not storagefilter.updateFlag:
            ds.dump(h5pathFileOut)

        LOGGER.logger.info(f"Trial {trialFilename} imported successfully in {h5pathFileOut}")

    



if __name__ == "__main__":

    main(args=None)
