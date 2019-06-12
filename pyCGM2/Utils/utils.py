# -*- coding: utf-8 -*-
def toBool(text):
    return True if text == "True" else False

def isInRange(val, min, max):

    if val<min or val>max:
        return False
    else:
        return True
