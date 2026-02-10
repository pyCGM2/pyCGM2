import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files

from pyCGM2.Mek.mek import mekTools

import matplotlib.pyplot as plt



def gatherCycle(data_group,label):
    count = 0
    list_of_arrays = []
    list_of_attrs = []
    for groupPath, set_obj in mekTools.iter_sets(data_group):
        print (f"Checking set in : {groupPath}")
        if label in groupPath and "Cycle" in groupPath:
            print(f"Set de {label} detected in : {groupPath}")
            setDetect = data_group.retrieve_set(data_group.name()+groupPath)
            name= setDetect.name()
            values = setDetect.read()
            list_of_arrays.append(values)
            attrs = {}
            for attrIt in setDetect.list_attributes_name():
                attrs[attrIt] = setDetect.retrieve_attribute(attrIt).read()
            list_of_attrs.append(attrs)

            count+=1
  
            


    # if is_list_of_arrays(list_of_arrays):
    #     flat = [x for sublist in list_of_arrays for x in sublist]
    # else:
    #      flat = list_of_arrays

    array_3d = np.stack(list_of_arrays, axis=0)

    return array_3d,list_of_attrs

