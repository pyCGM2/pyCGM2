
import pyCGM2
LOGGER = pyCGM2.LOGGER


# import sys
# from pyCGM2.Tools import btkTools
# import yaml



# Import optionnel de moveck
try:
    import moveck
    MOVECK_AVAILABLE = True
except ImportError:
    MOVECK_AVAILABLE = False
    moveck = None  # pour éviter un NameError plus tard
    LOGGER.logger.warning("moveck pipe is not installed")


if MOVECK_AVAILABLE:
    class Storage(object):
        """

        """
        def __init__(self, storagePathFile = None, updateFlag=False):

            self.updateFlag = False
            if storagePathFile is None:
                self.ds = moveck.data_store()
            else:
                if updateFlag :
                    self.ds = moveck.data_store(storagePathFile,moveck.data_store.update)
                else:
                    self.ds = moveck.data_store(storagePathFile)



        def getStorage(self):
            return self.ds




