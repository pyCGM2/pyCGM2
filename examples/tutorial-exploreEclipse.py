from os.path import dirname as up

from pyCGM2.Nexus import eclipse
from pyCGM2.Nexus import vskTools
from pyCGM2 import enums



path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-flow-data\\Nantes\\MAIGNAN Olympe\\Session 1\\"


# read a patient enf file
patientDir = up(up(path))+"\\"
enfPatientFile = eclipse.getEnfFiles(patientDir,enums.EclipseType.Patient)
enfPatient = eclipse.PatientEnfReader(patientDir,enfPatientFile)


enfPatient.m_patientInfos.getPatientInfos() # retrieve all info
enfPatient.m_patientInfos.get("PatientID") # retr

# read a session enf file

enfSessionFile = eclipse.getEnfFiles(patientDir,enums.EclipseType.Session)
enfSession = eclipse.PatientEnfReader(patientDir,enfSessionFile)


enfs = eclipse.getEnfFiles(path,enums.EclipseType.Trial)
trials=[]
for enf in enfs:
    enfTrial = eclipse.TrialEnfReader(path,enf)
    if enfTrial.get("TrialType") == "Motion":


trial = eclipse.TrialEnfReader(path,"PN01OP01S01SS03.Trial.enf")
trial.setForcePlates("XXX")
trial.save()

