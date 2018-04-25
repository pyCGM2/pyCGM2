# -*- coding: utf-8 -*-
import sys
import logging
import _winreg as winreg


PATH = 'Applications\python.exe\shell\open\command'
VALUE = '"PYEXE" "%1" %*'

def checkPythonRegistryKeyInRegistry():
    classes_root = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "",0,winreg.KEY_SET_VALUE)
    try:
        winreg.OpenKey(classes_root, PATH)
        return True
    except WindowsError:
        return False


def createPythonExeRegisterKey(pythonExe):

    classes_root = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "",0,winreg.KEY_SET_VALUE)

    key = winreg.CreateKey(classes_root, PATH)
    winreg.SetValue(key, '', winreg.REG_SZ, VALUE.replace('PYEXE', pythonExe))


def compatiblePythonExeRegisterKey(COMPATIBLENEXUSKEY):
    reg_key = winreg.OpenKey(
        winreg.HKEY_CLASSES_ROOT,
        PATH,0,winreg.KEY_ALL_ACCESS)

    value = winreg.QueryValue(reg_key, "")

    return True if value == COMPATIBLENEXUSKEY else False


# obsolete
# def getPythonExeRegisterKey():
#     reg_key = winreg.OpenKey(
#         winreg.HKEY_CLASSES_ROOT,
#         r'Applications\python.exe\shell\open\command',0,winreg.KEY_ALL_ACCESS)
#     value = winreg.QueryValue(reg_key, "")
#     return value

def setPythonExeRegisterKey(COMPATIBLENEXUSKEY):

    reg_key = winreg.OpenKey(
        winreg.HKEY_CLASSES_ROOT,
        PATH,0,winreg.KEY_ALL_ACCESS)

    winreg.SetValueEx(reg_key, "" ,0, winreg.REG_SZ, COMPATIBLENEXUSKEY)
