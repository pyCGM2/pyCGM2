# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:00:58 2016

@author: AAA34169
"""


import btk
import pyCGM2

from pyCGM2.Core.Tools import  btkTools



if __name__ == "__main__":


    # inputs
    newSubjectlabel = "CGM2"
    DATA_PATH =   "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGA-Data\\CGM1\\PIG standard\\basic\\"  
    filename = "MRI-US-01, 2008-08-08, 3DGA 14.c3d"    

    
    acq = btkTools.smartReader(str(DATA_PATH + filename))
    btkTools.modifySubject(acq,newSubjectlabel)
    btkTools.modifyEventSubject(acq,newSubjectlabel)
    btkTools.smartWriter(acq,str(DATA_PATH + filename))