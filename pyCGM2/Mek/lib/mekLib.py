"""Helper functions reading and writing *mek* (moveck) data stores."""
import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Mek.mek import mekTools
import yaml

def setDictToYamlAttribute(ds, dictIn, groupname,label): #"flow-userSettings"
    """Store a dictionary as a YAML-formatted attribute of a group.

    Args:
        ds: moveck data store.
        dictIn (dict): Dictionary to store.
        groupname (str): Path of the group.
        label (str): Attribute name (e.g. ``"flow-userSettings"``).
    """
    content = yaml.dump(dictIn, allow_unicode=True)
    group = ds.root().retrieve_group(groupname)
    group.create_attribute(label, content)


def readYamlAttribute(ds, groupname,label):
    """Read a YAML-formatted attribute of a group.

    Args:
        ds: moveck data store.
        groupname (str): Path of the group.
        label (str): Attribute name.

    Returns:
        dict: The decoded attribute.
    """
    group = ds.root().retrieve_group(groupname)
    settingRaw = group.retrieve_attribute(label).read()
    settings_dict = yaml.safe_load(settingRaw)
    return settings_dict



def gatherCycles(data_group,label):
    """Gather all cycle datasets of a label found in a group (searched recursively).

    Args:
        data_group: moveck group.
        label (str): Label of the variable (the dataset path must contain the label and ``Cycle``).

    Returns:
        Tuple[np.ndarray, List[dict]]: Stacked cycle values (cycles x frames x components)
        and the attributes of each cycle dataset.
    """
    count = 0
    list_of_arrays = []
    list_of_attrs = []
    for groupPath, set_obj in mekTools.iter_sets(data_group):
        # print (f"Checking set in : {groupPath}")
        if label in groupPath and "Cycle" in groupPath:
            # print(f"Set de {label} detected in : {groupPath}")
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

