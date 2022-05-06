# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pyCGM2; LOGGER = pyCGM2.LOGGER

from pyCGM2.Signal.detect_peaks import detect_peaks
from pyCGM2.Math import derivation

from pyCGM2.Processing import analysisHandler
import re

class JointPatternFilter(object):
    """
    """

    def __init__(self, jointPatternProcedure, analysis):

        self.m_procedure = jointPatternProcedure
        self.m_analysis = analysis

    @classmethod
    def interpretCriteria(cls,criteria):
        primaries=list()
        secondaries=list()

        criterias = criteria.split("+")
        for it in criterias:
            if  "(" not in it:
                status = re.findall( "\[(.*?)\]" ,it )[0]
                ide = re.findall("\#(\d{0,100})",it)[0]
                primaries.append({'status': status, 'ide': int(ide)})
            else:
                bracket =  re.findall( "\((.*?)\)" ,it )[0]
                score = re.findall(",(\d{0,100})", bracket)[0]
                dict2 = {"score":int(score), "case" : list()}
                casesStr = re.findall("(.*?),",bracket)[0]
                cases = casesStr.split("|")
                for case in cases:
                    status = re.findall( "\[(.*?)\]" ,case )[0]
                    ide = re.findall("\#(\d{0,100})",case)[0]
                    dict2["case"].append( {'status': status, 'ide': int(ide)})
                secondaries.append(dict2)
        return primaries, secondaries


    def getValues(self):
        dataframeData = self.m_procedure.detectValue(self.m_analysis)
        return dataframeData

    def getPatterns(self,filter=False):
        data = self.m_procedure.detectValue(self.m_analysis)
        dataframePattern = self.m_procedure.detectPattern()

        if not filter:
            return dataframePattern
        else:

            filteredDataPattern = dataframePattern.loc[dataframePattern['Success'].isin(['TRUE','partial-primary','partial-secondary'])]

            return filteredDataPattern#[['Context','Plane' ,'Pattern']]
