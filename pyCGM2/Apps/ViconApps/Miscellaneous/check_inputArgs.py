# -*- coding: utf-8 -*-
from __future__ import unicode_literals

"""
"""
#import ipdb
import logging
import argparse

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)



def main():


    parser = argparse.ArgumentParser(description='CGM1 Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix' )
    parser.add_argument('--resetMP', action='store_true', help='reset optional mass parameters')

    args = parser.parse_args()

    print (args)


if __name__ == "__main__":

    # ---- main script -----
    main()
