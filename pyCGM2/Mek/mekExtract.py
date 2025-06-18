import sys
from pyCGM2.Tools import btkTools



class mekStorageFilter(object):
    """

    """

    def __init__(self,mekStorage):
        self.m_storage = mekStorage.root()


    def run(self,scheme):
        extractGrp = self.m_storage.create_group("Extraction")
        for key in scheme:
            extractGrp.create_group(key)
            
            if scheme[key] != []:

                pathfilenames = scheme[key][0]
                targets = scheme[key][1]

                for pathfilename in pathfilenames:

                    filename = pathfilename.split("\\")[-1]
                    acq = btkTools.smartReader(pathfilename)

                    extractGrp.create_set(f"{key}/{filename}/events/Foot Strike/Left" ,[frame/acq.GetPointFrequency() for frame in btkTools.smartGetEvents(acq,"Foot Strike","Left") ] )
                    extractGrp.create_set(f"{key}/{filename}/events/Foot Off/Left" ,[frame/acq.GetPointFrequency() for frame in btkTools.smartGetEvents(acq,"Foot Off","Left") ] )
                    extractGrp.create_set(f"{key}/{filename}/events/Foot Strike/Right" ,[frame/acq.GetPointFrequency() for frame in btkTools.smartGetEvents(acq,"Foot Strike","Right") ] )
                    extractGrp.create_set(f"{key}/{filename}/events/Foot Off/Right" ,[frame/acq.GetPointFrequency() for frame in btkTools.smartGetEvents(acq,"Foot Off","Right") ] )

                    for target in targets:
                        variableName = target.split(".")[0]
                        group_name = f"{key}/{filename}/{variableName}"
                        values = acq.GetPoint(variableName).GetValues()
                        extractGrp.create_set(group_name ,values )
                        extractGrp.retrieve_set(group_name).create_attribute("StartTime",  acq.GetFirstFrame()/acq.GetPointFrequency())
                        extractGrp.retrieve_set(group_name).create_attribute("SampleRate",  acq.GetPointFrequency())
