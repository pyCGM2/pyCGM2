# -*- coding: utf-8 -*-
import os
from pyCGM2.Nexus import eclipse
from pyCGM2.Nexus import vskTools
from pyCGM2 import enums
from pyCGM2.flow.Nantes import eclipseInterface

class AbstractFlowProcedure(object):
    """abstract procedure """
    def __init__(self):
        pass
    def run(self):
        pass


class EclipseFlowProcedure(AbstractFlowProcedure):
    """    """
    def __init__(self):
        super(EclipseFlowProcedure, self).__init__()
 
    def run(self,data_path, modelVersion):

        modelVersion = modelVersion.replace(".","")
        
        parent = os.path.abspath(os.path.join(data_path,os.pardir))+"\\"

        vskFile = vskTools.getVskFiles(data_path)
        vsk = vskTools.Vsk(str(data_path +  vskFile))
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)


        eclipseInterface.repareEnf(data_path)

        patientEnfFile =  eclipse.getEnfFiles(parent,enums.EclipseType.Patient)
        patientInfo = eclipse.PatientEnfReader(parent,patientEnfFile)

        patient = dict()
        patient["PatientID"] = patientInfo.get("PatientID")

        sessionEnfFile =  eclipse.getEnfFiles(data_path,enums.EclipseType.Session)
        sessionInfo = eclipse.SessionEnfReader(data_path,sessionEnfFile)

        visit = dict()
        visit["Age"] = sessionInfo.get("Age")
        visit["SessionID"] = sessionInfo.get("SessionID")

        eclipseDate = sessionInfo.get("CREATIONDATEANDTIME").split(",")
        visit["Date"] = str(eclipseDate[2]) +"-"+ str(eclipseDate[1]) +"-"+ str(eclipseDate[0])

        staticTrials = eclipseInterface.findStaticTrials(data_path)

        calibs = eclipseInterface.staticDetails(data_path, staticTrials)

        motionTrials = eclipseInterface.findMotionTrials(data_path)
        emgTrials = eclipseInterface.findEmgTrials(data_path)
        mvcTrials = eclipseInterface.findMVCTrials(data_path)


        fits = eclipseInterface.FittingDetails(data_path,motionTrials)
        emgs = eclipseInterface.EmgDetails(data_path,emgTrials)
        mvcs = eclipseInterface.MvcDetails(data_path,mvcTrials)
        conditions = eclipseInterface.getConditions(data_path,motionTrials+emgTrials)


        data = {"ModelVersion":modelVersion,
                "Patient":patient,
                "Visit":visit,
                "Mp": required_mp,
                "Calibration":calibs,
                "Fitting":fits,
                "Emg":emgs,
                "Mvc":mvcs,
                "Conditions":conditions
                }
        
        return data



