import sys
from pyCGM2.Tools import btkTools
import numpy as np
from scipy.interpolate import interp1d

from pyCGM2.Mek.mek import mekTools
from pyCGM2.Cycles import cycleUtils
from pyCGM2.Cycles import cycleBuilders



import pyCGM2
LOGGER = pyCGM2.LOGGER

def normalize(values, attrs,eventStartTime, eventEndTime):
    
    initial_time = attrs["StartTime"]
    fs = attrs["SampleRate"]
    start_time = eventStartTime
    end_time = eventEndTime

    
    ncol = values.shape[1]
   
    # Convertir temps en indices
    start_idx = int((start_time - initial_time) * fs)
    end_idx = int((end_time - initial_time) * fs)

    # Extraction du segment
    segment = values[start_idx:end_idx+1]  # shape (M, 3)
    
    # Vecteur temps original du segment
    t_original = np.linspace(start_time, end_time, segment.shape[0])

    # Vecteur temps normalisé (101 points entre 0% et 100%)
    t_normalized = np.linspace(start_time, end_time, 101)


    # Interpolation colonne par colonne
    segment_normalized = np.zeros((101, ncol))
    for i in range(ncol):
        try:
            f = interp1d(t_original, segment[:, i], kind='linear')
        except:
            import ipdb; ipdb.set_trace()
        segment_normalized[:, i] = f(t_normalized)

    return segment_normalized


class mekNormalizeFilter(object):
    """

    """

    def __init__(self,mekStorage, group):
        self.m_storage = mekStorage.root()
        self.m_group = group



    def run(self,scheme,cropToForcePlateGroups=[]):
        
        if self.m_group is None:
            extractGrp = self.m_storage.retrieve_group("Extraction")
            normalizeGrp = self.m_storage.create_group("Normalize")
        else: 
            extractGrp = self.m_storage.retrieve_group(f"{self.m_group}/Extraction")

            normalizeGrp = self.m_storage.create_group(f"{self.m_group}/Normalize")

        
        for key in scheme:
            normalizeGrp.create_group(key)

            exist = True
            try: 
                extractGrp.retrieve_group(key)
            except RuntimeError:
                LOGGER.logger.warning(f"[pyCGM2] - No data extracted for {key}, skipping normalization")
                exist = False
                continue

            if scheme[key] != [] and exist:

                pathfilenames = scheme[key][0]
                targets = scheme[key][1]

                

                for pathfilename in pathfilenames:

                    filename = pathfilename.split("\\")[-1]

                    eventGr = extractGrp.retrieve_group(f"{key}/{filename}/events")  

                    for target in targets:
                        if "=" in target:
                            targetName = target.split("=")[0].split(":")[0]
                            eventContext = target.split("=")[0].split(":")[1]
                            variableName = target.split("=")[1]
                        else:
                            targetName = target.split(":")[0]
                            eventContext = target.split(":")[1]
                            variableName = targetName 

                        
                        set_name = f"{key}/{filename}/{variableName}"
                        if extractGrp.exists_set(set_name):
                            data = extractGrp.retrieve_set(set_name)
                            attrs = mekTools.mekAttributesToDict(data)


                            if eventContext == "Left":
                                events = eventGr.retrieve_set("Foot Strike/Left").read()
                                eventsfo = eventGr.retrieve_set("Foot Off/Left").read()
                                
                                eventsfo_opp = eventGr.retrieve_set("Foot Off/Right").read()
                                eventsfs_opp = eventGr.retrieve_set("Foot Strike/Right").read()

 

                                cycles = cycleBuilders.build_cycles_fromEvents(events, eventsfo, eventsfs_opp, eventsfo_opp)
                                try:
                                    fp_events = eventGr.retrieve_set("ForcePlateEvents/Left").read()
                                except:
                                    fp_events = None


                            elif eventContext == "Right":
    
                                events = eventGr.retrieve_set("Foot Strike/Right").read()
                                eventsfo = eventGr.retrieve_set("Foot Off/Right").read()
                                try:
                                    fp_events = eventGr.retrieve_set("ForcePlateEvents/Right").read()
                                except:
                                    fp_events = None

                                eventsfo_opp = eventGr.retrieve_set("Foot Off/Left").read()
                                eventsfs_opp = eventGr.retrieve_set("Foot Strike/Left").read()


                                cycles = cycleBuilders.build_cycles_fromEvents(events, eventsfo, eventsfs_opp, eventsfo_opp)

                            ncycles = len(cycles)


                            if any(name in key for name in cropToForcePlateGroups) and fp_events is not None:
                                for fpevent in fp_events:
                                    index=0
                                    for cycle in cycles:
                                        if fpevent >= cycle["start"] and fpevent <= cycle["end"]:
                                            cycleValues = normalize(data.read(),attrs,cycle["start"],cycle["end"])
                                            group_name = f"{key}/{filename}/{variableName}/Cycle{index}"
                                            if normalizeGrp.exists_set( group_name):
                                                normalizeGrp.retrieve_set(group_name).write(cycleValues)
                                            else:
                                                normalizeGrp.create_set(group_name ,cycleValues )
                                            
                                            footOff = (cycle["footOff"]-cycle["start"])/( cycle["end"]-cycle["start"])*100
                                            normalizeGrp.retrieve_set(group_name).create_attribute("footOff",  round(footOff))

                                            controlateralFootOff = (cycle["controlateral_footOff"]-cycle["start"])/( cycle["end"]-cycle["start"])*100
                                            normalizeGrp.retrieve_set(group_name).create_attribute("controlateral_footOff",  round(controlateralFootOff))

                                            controlateralFootStrike = (cycle["controlateral_footStrike"]-cycle["start"])/( cycle["end"]-cycle["start"])*100
                                            normalizeGrp.retrieve_set(group_name).create_attribute("controlateral_footStrike",  round(controlateralFootStrike))

                                        index+=1
                                    
                            else:
                                index=0
                                for cycle in cycles:
                                    cycleValues = normalize(data.read(),attrs,cycle["start"],cycle["end"])
                                    group_name = f"{key}/{filename}/{variableName}/Cycle{index}"
                                    
                                    if normalizeGrp.exists_set( group_name):
                                        normalizeGrp.retrieve_set(group_name).write(cycleValues)
                                    else:
                                        normalizeGrp.create_set(group_name ,cycleValues )
                                    

                                    footOff = (cycle["footOff"]-cycle["start"])/( cycle["end"]-cycle["start"])*100
                                    normalizeGrp.retrieve_set(group_name).create_attribute("footOff",  round(footOff))

                                    controlateralFootOff = (cycle["controlateral_footOff"]-cycle["start"])/( cycle["end"]-cycle["start"])*100
                                    normalizeGrp.retrieve_set(group_name).create_attribute("controlateral_footOff",  round(controlateralFootOff))

                                    controlateralFootStrike = (cycle["controlateral_footStrike"]-cycle["start"])/( cycle["end"]-cycle["start"])*100
                                    normalizeGrp.retrieve_set(group_name).create_attribute("controlateral_footStrike",  round(controlateralFootStrike))
                                    
                                    index+=1





        

