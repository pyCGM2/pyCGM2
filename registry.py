# -*- coding: utf-8 -*-
import sys

import _winreg as winreg

def getPythonExeRegisterKey():
    reg_key = winreg.OpenKey(
        winreg.HKEY_CLASSES_ROOT,
        r'Applications\python.exe\shell\open\command',0,winreg.KEY_ALL_ACCESS)


    value = winreg.QueryValue(reg_key, "")
    
    return value
    
def setPythonExeRegisterKey(COMPATIBLENEXUSKEY):
    
    reg_key = winreg.OpenKey(
        winreg.HKEY_CLASSES_ROOT,
        r'Applications\python.exe\shell\open\command',0,winreg.KEY_ALL_ACCESS)

    winreg.SetValueEx(reg_key, "" ,0, winreg.REG_SZ, COMPATIBLENEXUSKEY)