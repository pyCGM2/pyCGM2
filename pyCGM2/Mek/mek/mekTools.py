
def mekAttributesToDict(group):
    out = dict()
    attrs = group.list_attributes_name()
    if attrs !=[]:
        for attr in attrs:
            out[attr] = group.retrieve_attribute(attr).read()
    return out

def writeDataset(group, name, data, attributes=None):
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
    # Yield les sets du groupe courant
    for nom_set in groupe.list_set_children_name():
        yield f"{prefix}/{nom_set}", groupe.retrieve_set(nom_set)

    # Parcours récursif des sous-groupes
    for nom_groupe in groupe.list_group_children_name():
        sous_groupe = groupe.retrieve_group(nom_groupe)
        nouveau_prefix = f"{prefix}/{nom_groupe}"
        yield from iter_sets(sous_groupe, nouveau_prefix)      

def iter_grp(groupe, prefix=""):
    # Yield les sets du groupe courant
    for nom_gr in groupe.list_group_children_name():
        yield f"{prefix}/{nom_gr}", groupe.retrieve_group(nom_gr)

    # Parcours récursif des sous-groupes
    for nom_groupe in groupe.list_group_children_name():
        sous_groupe = groupe.retrieve_group(nom_groupe)
        nouveau_prefix = f"{prefix}/{nom_groupe}"
        yield from iter_grp(sous_groupe, nouveau_prefix) 

         