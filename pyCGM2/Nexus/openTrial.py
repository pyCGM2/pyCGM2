# -*- coding: utf-8 -*-
"""
convenient app for calling python nexus "OpenTrial" command.

This app need full filename without extension as input argument. 

**usage**

  - from dos : python "opentrial.py c:/.../ static 01"  
  

  

"""

import os
import sys
#import logging
#import argparse

import pdb

# pyCGM2 settings
import pyCGM2
#pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
pyCGM2.CONFIG.addNexusPythonSdk()
import ViconNexus

if __name__ == "__main__":
    
    pyNEXUS = ViconNexus.ViconNexus()    
    NEXUS_PYTHON_CONNECTED = pyNEXUS.Client.IsConnected()

#    NEXUS_PYTHON_CONNECTED = True   
     
    if NEXUS_PYTHON_CONNECTED: # run Operation
    
        print "Call.... Nexus"
        print str(sys.argv[1])
        pyNEXUS.OpenTrial( str(sys.argv[1]), 30 )    
    