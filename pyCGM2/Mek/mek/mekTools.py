"""Low-level tools handling moveck groups and datasets."""

def mekAttributesToDict(group):
    """Return the attributes of a group (or dataset) as a dictionary.

    Args:
        group: moveck group or dataset.

    Returns:
        dict: Attribute values by name.
    """
    out = dict()
    attrs = group.list_attributes_name()
    if attrs !=[]:
        for attr in attrs:
            out[attr] = group.retrieve_attribute(attr).read()
    return out

def writeDataset(group, name, data, attributes=None):
    """Write a dataset in a group, creating it if needed.

    Args:
        group: moveck group.
        name (str): Dataset name.
        data: Values to write.
        attributes (dict, optional): Attributes to create on the dataset.
    """
    if group.exists_set(name):
        group.retrieve_set(name).write(data)
        if attributes is not None:
            for attr in attributes:
                group.retrieve_set(name).create_attribute(attr, attributes[attr])

    else:
        group.create_set(name, data)
        if attributes is not None:
            for attr in attributes:
                group.retrieve_set(name).create_attribute(attr, attributes[attr])    




def iter_sets(groupe, prefix=""):
    """Iterate recursively over the datasets of a group.

    Args:
        groupe: moveck group.
        prefix (str, optional): Path prefix. Defaults to "".

    Yields:
        Tuple[str, object]: Relative path and dataset.
    """
    # Yield les sets du groupe courant
    for nom_set in groupe.list_set_children_name():
        yield f"{prefix}/{nom_set}", groupe.retrieve_set(nom_set)

    # Parcours récursif des sous-groupes
    for nom_groupe in groupe.list_group_children_name():
        sous_groupe = groupe.retrieve_group(nom_groupe)
        nouveau_prefix = f"{prefix}/{nom_groupe}"
        yield from iter_sets(sous_groupe, nouveau_prefix)      

def iter_grp(groupe, prefix=""):
    """Iterate recursively over the sub-groups of a group.

    Args:
        groupe: moveck group.
        prefix (str, optional): Path prefix. Defaults to "".

    Yields:
        Tuple[str, object]: Relative path and group.
    """
    # Yield les sets du groupe courant
    for nom_gr in groupe.list_group_children_name():
        yield f"{prefix}/{nom_gr}", groupe.retrieve_group(nom_gr)

    # Parcours récursif des sous-groupes
    for nom_groupe in groupe.list_group_children_name():
        sous_groupe = groupe.retrieve_group(nom_groupe)
        nouveau_prefix = f"{prefix}/{nom_groupe}"
        yield from iter_grp(sous_groupe, nouveau_prefix) 

         