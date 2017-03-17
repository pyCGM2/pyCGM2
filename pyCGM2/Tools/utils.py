# -*- coding: utf-8 -*-
import os
import pdb


def getFiles(path, extension, ignore=None):

    out=list()
    for file in os.listdir(path):
        if ignore is None:
            if file.endswith(extension):
                out.append(file)
        else:
            if file.endswith(extension) and ignore not in file:
                out.append(file)
    
    return out
    