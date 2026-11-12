"""Initialisation of *mek* data stores.

The moveck package is optional: the :class:`Storage` class is only defined if moveck is installed.
"""

import pyCGM2
LOGGER = pyCGM2.LOGGER
import os

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
        """Wrapper around a moveck data store.

        Args:
            storagePathFile (str, optional): Path of an existing data store. If None, a new
                in-memory data store is created.
            updateFlag (bool, optional): Open the existing data store in update mode. Defaults to False.

        Raises:
            Exception: If the file does not exist or is empty.
        """
        def __init__(self, storagePathFile = None, updateFlag=False):
            self.updateFlag = updateFlag
            if storagePathFile is None:
                self.ds = moveck.data_store()
            else:
                if not os.path.exists(storagePathFile):
                    LOGGER.logger.error(f"the file ({storagePathFile} not exist")
                    raise Exception("file not exist")

                if updateFlag :
                    self.ds = moveck.data_store(storagePathFile,moveck.data_store.update)
                else:
                    self.ds = moveck.data_store(storagePathFile)
                
                if self.ds.root().list_group_children_name()==[]:
                    LOGGER.logger.error(f"file ({storagePathFile} called but content is empty")
                    raise Exception("empty")




        def getStorage(self):
            """Return the moveck data store.

            Returns:
                moveck.data_store: The data store.
            """
            return self.ds




