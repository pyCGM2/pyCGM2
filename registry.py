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


def getPythonExeRegisterKey2():

    try:
        reg_key = winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT,
            r'Applications\python2.exe\shell\open\command',0,winreg.KEY_ALL_ACCESS)

    value = winreg.QueryValue(reg_key, "")

    return value


# def createKey(COMPATIBLENEXUSKEY):
# don t work !!
#
#     key = wreg.CreateKey(wreg.HKEY_CLASSES_ROOT, "Applications\\python2.exe\\shell\\open\\command")
#     winreg.SetValueEx(reg_key, "" ,0, winreg.REG_SZ, COMPATIBLENEXUSKEY)
