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

    def __init__(self,group):
        self.m_group = group
        

    def run(self):

        ds = moveck.data_store()
        root = ds.root()
        root.create_group(self.m_group)
        return ds
