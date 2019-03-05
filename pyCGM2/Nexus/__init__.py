import os
import sys

def getLastNexusVersion():
    nexusDir = "C:\Program Files (x86)\Vicon"
    dirs = os.listdir(nexusDir)
    li =[]
    for it in dirs:
        if "Nexus2" in it:
            version = int(it[it.find(".")+1:])
            li.append(version)
    last = max(li)
    return "Nexus2."+str(last)


NEXUS_VERSION = getLastNexusVersion()
if not "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Python" in sys.path:
    sys.path.append( "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Python")

if not "C:/Program Files (x86)/Vicon/Nexus"+NEXUS_VERSION+"/SDK/Win32" in sys.path:
    sys.path.append( "C:/Program Files (x86)/Vicon/"+NEXUS_VERSION+"/SDK/Win32")
