import sys
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools

import pyCGM2

# Import optionnel de moveck
try:
    import moveck
    MOVECK_AVAILABLE = True
except ImportError:
    MOVECK_AVAILABLE = False
    moveck = None  # pour éviter un NameError plus tard
    LOGGER.warning("moveck pipe is not installed")

if MOVECK_AVAILABLE:
    class mekInitStorageFilter(object):
        """

        """

        def __init__(self, storagePathFile = None):

            self.updateFlag = False
            if storagePathFile is None:
                self.ds = moveck.data_store()
            else:
                self.ds = moveck.data_store(storagePathFile,moveck.data_store.update)
                self.updateFlag = True


                self.ds.root().list_group_children_name()

            
        
        def getStorage(self):
            return self.ds
