# -*- coding: utf-8 -*-
import pyCGM2
import shutil


def main():

    initFolder = pyCGM2.MAIN_PYCGM2_PATH+"pyCGM2\\Settings"
    appDataFolder = pyCGM2.PYCGM2_APPDATA_PATH[:-1]

    shutil.copytree(initFolder, appDataFolder)

if __name__ == "__main__":

    main()
