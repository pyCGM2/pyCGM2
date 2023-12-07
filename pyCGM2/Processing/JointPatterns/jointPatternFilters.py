# -*- coding: utf-8 -*-
"""
This module provides a filter for analyzing and interpreting joint patterns in motion data. 
It allows for the detection of specific patterns in joint movements using defined procedures 
and can be used to filter and extract relevant information from these analyses.

"""
import numpy as np
import pandas as pd
import pyCGM2; LOGGER = pyCGM2.LOGGER


import re

from pyCGM2.Processing.JointPatterns.jointPatternProcedures import JointPatternProcedure
from pyCGM2.Processing.analysis import Analysis
from typing import List, Tuple, Dict, Optional,Union,Any

class JointPatternFilter(object):
    """
    Filter for analyzing and interpreting joint patterns in movement analyses.

    Args:
        jointPatternProcedure (JointPatternProcedure): The procedure used for joint pattern detection.
        analysis (Analysis): The analysis instance containing the data to be analyzed.
    """

    def __init__(self, jointPatternProcedure:JointPatternProcedure, analysis:Analysis):

        self.m_procedure = jointPatternProcedure
        self.m_analysis = analysis

    @classmethod
    def interpretCriteria(cls,criteria:str):
        """
        Interprets criteria for joint pattern models.

        Args:
            criteria (str): Criteria defined for pattern detection.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Returns primary and secondary criteria.
        """
        primaries=[]
        secondaries=[]

        criterias = criteria.split("+")
        for it in criterias:
            if  "(" not in it:
                status = re.findall( "\[(.*?)\]" ,it )[0]
                ide = re.findall("\#(\d{0,100})",it)[0]
                primaries.append({'status': status, 'ide': int(ide)})
            else:
                bracket =  re.findall( "\((.*?)\)" ,it )[0]
                score = re.findall(",(\d{0,100})", bracket)[0]
                dict2 = {"score":int(score), "case" : []}
                casesStr = re.findall("(.*?),",bracket)[0]
                cases = casesStr.split("|")
                for case in cases:
                    status = re.findall( "\[(.*?)\]" ,case )[0]
                    ide = re.findall("\#(\d{0,100})",case)[0]
                    dict2["case"].append( {'status': status, 'ide': int(ide)})
                secondaries.append(dict2)
        return primaries, secondaries


    def getValues(self):
        """
        Extracts values from detected joint patterns.

        Returns:
            pd.DataFrame: DataFrame containing the extracted values.
        """
        dataframeData = self.m_procedure.detectValue(self.m_analysis)
        return dataframeData

    def getPatterns(self,filter:bool=False):
        """
        Detects and returns joint patterns.

        Args:
            filter (bool, optional): If True, filters the results. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame of detected joint patterns.
        """

        data = self.m_procedure.detectValue(self.m_analysis)
        dataframePattern = self.m_procedure.detectPattern()

        if not filter:
            return dataframePattern
        else:

            filteredDataPattern = dataframePattern.loc[dataframePattern['Success'].isin(['TRUE','partial-primary','partial-secondary'])]

            return filteredDataPattern#[['Context','Plane' ,'Pattern']]
