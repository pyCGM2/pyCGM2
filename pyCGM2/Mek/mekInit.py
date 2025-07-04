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

        def __init__(self):
            self.ds = moveck.data_store()
            
            

        def createGroup(self, group):
            self.ds.root().create_group(group)
        
        def getStorage(self):
            return self.ds
