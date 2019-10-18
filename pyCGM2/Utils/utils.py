# -*- coding: utf-8 -*-
import pyCGM2

def toBool(text):
    return True if text == "True" else False

def isInRange(val, min, max):

    if val<min or val>max:
        return False
    else:
        return True

def str(unicodeVariable):
    return unicodeVariable.encode(pyCGM2.ENCODER)
