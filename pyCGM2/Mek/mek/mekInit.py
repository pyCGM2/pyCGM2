
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
    LOGGER.warning("moveck pipe is not installed")


if MOVECK_AVAILABLE:
    class Storage(object):
        """

        """
        def __init__(self, storagePathFile = None):

            self.updateFlag = False
            if storagePathFile is None:
                self.ds = moveck.data_store()
            else:
                self.ds = moveck.data_store(storagePathFile,moveck.data_store.update)
                self.updateFlag = True


        def getStorage(self):
            return self.ds




