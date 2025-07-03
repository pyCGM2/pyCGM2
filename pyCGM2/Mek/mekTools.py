from collections import OrderedDict




def save_settings_text_to_h5(settings_path, h5file):
    """
    Lit un fichier texte contenant les settings et l'enregistre dans un groupe 'Settings'
    d'un fichier HDF5 ouvert via une API personnalisée.

    Parameters:
    - settings_path: chemin du fichier texte (ex: 'CGM23.userSettings')
    - h5file: instance d'un gestionnaire HDF5 (doit déjà être ouvert en écriture)
    """
    # Lire le contenu brut du fichier settings
    with open(settings_path, "r", encoding="utf-8") as f:
        settings_text = f.read()

    # Créer le groupe Settings et enregistrer le contenu comme dataset
    settings_group = h5file.create_group("Settings")
    settings_group.create_dataset("SettingsRaw", data=settings_text)





def save_settings_to_h5_group(group, dictionary):
    for key, value in dictionary.items():
        clean_key = str(key)

        if isinstance(value, (dict, OrderedDict)):
            subgroup = group.create_group(clean_key)
            save_settings_to_h5_group(subgroup, value)

        elif isinstance(value, list):
            if all(isinstance(v, (dict, OrderedDict)) for v in value):
                # Cas particulier : liste de dictionnaires (ex. section Fitting)
                list_group = group.create_group(clean_key)
                for i, item in enumerate(value):
                    item_group = list_group.create_group(f"Item_{i}")
                    save_settings_to_h5_group(item_group, item)
            else:
                # Liste "simple" (int, float, str...) => attribut
                group.create_attribute(clean_key, value)

        else:
            attr_value = value if value is not None else ""
            group.create_attribute(clean_key, attr_value) 




def write_dict_to_hdf5(hdf5_group, dictionary):
    for key, value in dictionary.items():
        # Nettoyage du nom pour HDF5 (pas d'espaces ou caractères spéciaux)
        clean_key = str(key).replace(" ", "_")

        if isinstance(value, (dict, OrderedDict)):
            subgroup = hdf5_group.create_group(clean_key)
            write_dict_to_hdf5(subgroup, value)
        else:
            # HDF5 ne supporte pas None -> remplacer par string vide ou "None"
            attr_value = value if value is not None else ""
            subgroup = hdf5_group
            subgroup.create_attribute(clean_key, attr_value)


def mekAttributesToDict(group):
    out = dict()
    attrs = group.list_attributes_name()
    if attrs !=[]:
        for attr in attrs:
            out[attr] = group.retrieve_attribute(attr).read()
    return out


def afficher_sets(groupe, prefix=""):
            # Lister les sets dans ce groupe
            sets = groupe.list_set_children_name()
            for nom_set in sets:
                print(f"Set : {prefix}/{nom_set}")

            # Lister les sous-groupes
            sous_groupes = groupe.list_group_children_name()
            for nom_groupe in sous_groupes:
                sous_groupe = groupe.retrieve_group(nom_groupe)
                nouveau_prefix = f"{prefix}/{nom_groupe}"
                afficher_sets(sous_groupe, nouveau_prefix)

def iter_sets(groupe, prefix=""):
    # Yield les sets du groupe courant
    for nom_set in groupe.list_set_children_name():
        yield f"{prefix}/{nom_set}", groupe.retrieve_set(nom_set)

    # Parcours récursif des sous-groupes
    for nom_groupe in groupe.list_group_children_name():
        sous_groupe = groupe.retrieve_group(nom_groupe)
        nouveau_prefix = f"{prefix}/{nom_groupe}"
        yield from iter_sets(sous_groupe, nouveau_prefix)        