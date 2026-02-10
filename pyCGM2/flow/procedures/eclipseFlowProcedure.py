# -*- coding: utf-8 -*-
import os
from pyCGM2.Nexus import eclipse
from pyCGM2.Nexus import vskTools
from pyCGM2 import enums
from pyCGM2.Nexus import eclipseFlowInterface

from pyCGM2.flow.procedures import flowProcedures


class EclipseFlowProcedure(flowProcedures.AbstractFlowProcedure):
    """    """
    def __init__(self):
        super(EclipseFlowProcedure, self).__init__()
 
    def run(self,data_path, modelVersion):

        # modelVersion = modelVersion.replace(".","")
        
        parent = os.path.abspath(os.path.join(data_path,os.pardir))+"\\"

        vskFile = vskTools.getVskFiles(data_path)
        vsk = vskTools.Vsk(str(data_path +  vskFile))
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)


        eclipseFlowInterface.repareEnf(data_path)

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

        staticTrials = eclipseFlowInterface.findStaticTrials(data_path)

        calibs = eclipseFlowInterface.staticDetails(data_path, staticTrials)

        motionTrials = eclipseFlowInterface.findMotionTrials(data_path)
        emgTrials = eclipseFlowInterface.findEmgTrials(data_path)
        mvcTrials = eclipseFlowInterface.findMVCTrials(data_path)


        fits = eclipseFlowInterface.FittingDetails(data_path,motionTrials)
        emgs = eclipseFlowInterface.EmgDetails(data_path,emgTrials)
        mvcs = eclipseFlowInterface.MvcDetails(data_path,mvcTrials)
        conditions = eclipseFlowInterface.getConditions(data_path,motionTrials+emgTrials)

        

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



