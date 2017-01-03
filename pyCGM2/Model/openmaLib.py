# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 21:37:48 2016

@author: aaa34169
"""

import ma.body


def renameOpenMAtoVicon(analysis, suffix=""):
    tss = analysis.child(0).findChildren(ma.T_TimeSequence)
    for ts in tss:
        name = ts.name()
        
        if "Angle" in name:
            newName = name.replace(".", "")
            newName= newName + "s" + suffix
            if ("Pelvis" in name):
                newName = newName.replace("Progress", "") + suffix
        if "Force" in name:
            newName = name.replace(".", "")
            newName = newName[0: newName.rfind("Force")+5] + suffix
        if "Moment" in name:
            newName = name.replace(".", "")
            newName = newName[0: newName.rfind("Moment")+6] + suffix
        if "Power" in name:
            newName = name.replace(".", "")
            newName = newName[0: newName.rfind("Power")+5] + suffix
        ts.setName(newName)