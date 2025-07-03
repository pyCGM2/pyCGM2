import sys
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools

MOVECKPATH = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\moveck\\"
sys.path.append(MOVECKPATH+"Moveck_pipe-2024.1.0-win64-pipeline_install\\packages")
import moveck

class mekInitStorageFilter(object):
    """

    """

    def __init__(self):
        self.ds = moveck.data_store()
        
        

    def createGroup(self, group):
        self.ds.root().create_group(group)
    
    def getStorage(self):
        return self.ds
