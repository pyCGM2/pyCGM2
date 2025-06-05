def mekAttributesToDict(group):
    out = dict()
    attrs = group.list_attributes_name()
    if attrs !=[]:
        for attr in attrs:
            out[attr] = group.retrieve_attribute(attr).read()
    return out



