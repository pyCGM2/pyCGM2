import sys
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools



class mekExtractFilter(object):
    """

    """

    def __init__(self,mekStorage,group=None):
        self.m_storage = mekStorage.root()
        self.m_group = group


    def run(self,scheme):

        if self.m_group is None:
          extractGrp = self.m_storage.create_group("Extraction")
        else : 
            extractGrp = self.m_storage.create_group(f"{self.m_group}/Extraction")

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
 
                    try:
                        extractGrp.create_set(f"{key}/{filename}/events/ForcePlateEvents/Left" ,[frame/acq.GetPointFrequency() for frame in btkTools.smartGetEvents(acq,"Left-FP","General") ] )
                    except:
                        pass
                    try:
                        extractGrp.create_set(f"{key}/{filename}/events/ForcePlateEvents/Right" ,[frame/acq.GetPointFrequency() for frame in btkTools.smartGetEvents(acq,"Right-FP","General") ] )
                    except:
                        pass


                    for target in targets:
                        if "=" in target:
                            targetName = target.split("=")[0].split(":")[0]
                            variableName = target.split("=")[1]
                        else:
                            targetName = target.split(":")[0]
                            variableName = targetName
                        
                        
                        

                        values = None
                        frequency = None

                        try:
                            values = acq.GetPoint(targetName).GetValues()
                            frequency = acq.GetPointFrequency()
                        except RuntimeError:
                            try:
                                values = acq.GetAnalog(targetName).GetValues()
                                frequency = acq.GetAnalogFrequency()
                            except RuntimeError:
                                LOGGER.logger.warning(f"[pyCGM2] - {targetName} not detected in {filename}")

                        if values is not None:
                            groupSet_name = f"{key}/{filename}/{variableName}"
                            LOGGER.logger.info(f"[pyCGM2] - Extracting {groupSet_name}")
                            extractGrp.create_set(groupSet_name, values)
                            extractGrp.retrieve_set(groupSet_name).create_attribute("StartTime", acq.GetFirstFrame() / acq.GetPointFrequency())
                            extractGrp.retrieve_set(groupSet_name).create_attribute("SampleRate", frequency )