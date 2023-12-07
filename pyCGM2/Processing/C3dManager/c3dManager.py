import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import List, Tuple, Dict, Optional,Union,Any

class C3dManager(object):
    """ 
    A C3dManager instance that organizes btk.Acquisition instances and associated filenames for different computational objectives.
    """
    def __init__ (self):
        self.spatioTemporal={"Acqs":None , "Filenames":None}
        self.kinematic={"Acqs":None , "Filenames":None}
        self.kineticFlag = False
        self.kinetic={"Acqs": None, "Filenames":None }
        self.emg={"Acqs":None , "Filenames":None}
        self.muscleGeometry={"Acqs": None , "Filenames":None}
        
        self.muscleDynamicFlag = False
        self.muscleDynamic={"Acqs":None , "Filenames":None}