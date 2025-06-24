import sys
from pyCGM2.Tools import btkTools
import numpy as np
from scipy.interpolate import interp1d

from pyCGM2.Mek import mekTools


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
        f = interp1d(t_original, segment[:, i], kind='linear')
        segment_normalized[:, i] = f(t_normalized)

    return segment_normalized


class mekNormalizeFilter(object):
    """

    """

    def __init__(self,mekStorage, group):
        self.m_storage = mekStorage.root()
        self.m_group = group

    def run(self,scheme):
        
        if self.m_group is None:
            extractGrp = self.m_storage.retrieve_group("Extraction")
            normalizeGrp = self.m_storage.create_group("Normalize")
        else: 
            extractGrp = self.m_storage.retrieve_group(f"{self.m_group}/Extraction")
            normalizeGrp = self.m_storage.create_group(f"{self.m_group}/Normalize")

        for key in scheme:
            normalizeGrp.create_group(key)

            if scheme[key] != []:

                pathfilenames = scheme[key][0]
                targets = scheme[key][1]

                for pathfilename in pathfilenames:

                    filename = pathfilename.split("\\")[-1]

                    eventGr = extractGrp.retrieve_group(f"{key}/{filename}/events")  

                    for target in targets:
                        variableName = target.split(".")[0]
                        eventContext = target.split(".")[1]
                        set_name = f"{key}/{filename}/{variableName}"
                        if extractGrp.exists_set(set_name):
                            data = extractGrp.retrieve_set(set_name)
                            attrs = mekTools.mekAttributesToDict(data)

                            if eventContext == "Left":
                                events = eventGr.retrieve_set("Foot Strike/Left").read()
                            elif eventContext == "Right":
                                events = eventGr.retrieve_set("Foot Strike/Right").read()

                            ncycles = len(events)-1

                            for i in range(ncycles):
                                cycleValues = normalize(data.read(),attrs,events[i],events[i+1])
                                group_name = f"{key}/{filename}/{variableName}/Cycle{i}"
                                normalizeGrp.create_set(group_name ,cycleValues )
                                normalizeGrp.retrieve_set(group_name).create_attribute("Valid",  True)





        

