# -*- coding: utf-8 -*-
import numpy as np
import ipdb
import pandas as pd
from pyCGM2.Tools import exportTools
from collections import OrderedDict
from pyCGM2.Signal.detect_peaks import detect_peaks
from pyCGM2.Math import derivation

import matplotlib.pyplot as plt
# --- FILTER ----


class AutomaticExtractionFilter(object):
    """
    """

    def __init__(self,xlsRules, analysis):

        self.m_analysis=analysis
        self.m_rules = pd.read_excel(xlsRules)

    def _buildPhases(self,dataStats):

        #phases
        phases = dict()

        # Left
        phases["Left","St"] = [0,int(dataStats.pst["stancePhase","Left"]["mean"])]
        phases["Left","Sw"] = [int(dataStats.pst["stancePhase","Left"]["mean"]),
                                        100]

        phases["Left","DS1"] = [0,
                                        int(dataStats.pst["doubleStance1","Left"]["mean"])]
        phases["Left","SS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"]) ]

        phases["Left","DS2"] = [int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"])]

        phases["Left","ISw"] = [int(dataStats.pst["stancePhase","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Left"]["mean"])]

        phases["Left","MSw"] = [int(dataStats.pst["stancePhase","Left"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Left"]["mean"]),
                                        int(dataStats.pst["stancePhase","Left"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Left"]["mean"])]

        phases["Left","TSw"] = [int(dataStats.pst["stancePhase","Left"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Left"]["mean"]),
                                        100]


        # Right
        phases["Right","St"] = [0,int(dataStats.pst["stancePhase","Right"]["mean"])]
        phases["Right","Sw"] = [int(dataStats.pst["stancePhase","Right"]["mean"]),
                                        100]

        phases["Right","DS1"] = [0,
                                        int(dataStats.pst["doubleStance1","Right"]["mean"])]
        phases["Right","SS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"]) ]

        phases["Right","DS2"] = [int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"])]

        phases["Right","ISw"] = [int(dataStats.pst["stancePhase","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Right"]["mean"])]

        phases["Right","MSw"] = [int(dataStats.pst["stancePhase","Right"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Right"]["mean"]),
                                        int(dataStats.pst["stancePhase","Right"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Right"]["mean"])]

        phases["Right","TSw"] = [int(dataStats.pst["stancePhase","Right"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Right"]["mean"]),
                                        100]




        return phases

    def _computeMethod(self, values, context, method, CyclePeriod, phases):

        if CyclePeriod == "GC":
            frameLimits= [0, 100]
        elif CyclePeriod == "St":
            frameLimits= phases[context,"St"]
        elif CyclePeriod == "Sw":
            frameLimits= phases[context,"Sw"]
        elif CyclePeriod == "DS1":
            frameLimits= phases[context,"DS1"]
        elif CyclePeriod == "SS":
            frameLimits= phases[context,"SS"]
        elif CyclePeriod == "DS2":
            frameLimits= phases[context,"DS2"]
        elif CyclePeriod == "ISw":
            frameLimits= phases[context,"ISw"]
        elif CyclePeriod == "MSw":
            frameLimits= phases[context,"MSw"]
        elif CyclePeriod == "TSw":
            frameLimits= phases[context,"TSw"]
        else:
            raise Exception ("Cycle period doesn t recognize")


        if method == "mean":
            val = np.mean(values[frameLimits[0]:frameLimits[1]])
        elif method == "range":
            val = np.abs( np.max(values[frameLimits[0]:frameLimits[1]]) -
                          np.min(values[frameLimits[0]:frameLimits[1]]))
        elif method == "min":
            val = np.min(values[frameLimits[0]:frameLimits[1]])
        elif method == "max":
            val = np.max(values[frameLimits[0]:frameLimits[1]])


        return val

    def _applyRules(self,value, row,ruleType):

        pass_rules = False
        comment = None

        if ruleType == 1:
            if value> row["Norm_Min"].values[0] and value<row["Norm_Max"].values[0]:
                comment = row["Comment_Normal"].values[0]
                pass_rules = True
            elif value> row["Norm_Max"].values[0] and value<row["Norm_Low"].values[0]:
                comment = row["Comment_Low"].values[0]
                pass_rules = True
            elif value> row["Norm_Low"].values[0] and value<row["Norm_Moderate"].values[0]:
                comment = row["Comment_Moderate"].values[0]
                pass_rules = True
            elif value> row["Norm_Moderate"].values[0] :
                comment = row["Comment_High"].values[0]
                pass_rules = True

        elif ruleType == 2:
            if value> row["Norm_Max"].values[0] and value<row["Norm_Min"].values[0]:
                comment = row["Comment_Normal"].values[0]
            elif value< row["Norm_Max"].values[0] and value>row["Norm_Low"].values[0]:
                comment = row["Comment_Low"].values[0]
            elif value< row["Norm_Low"].values[0] and value>row["Norm_Moderate"].values[0]:
                comment = row["Comment_Moderate"].values[0]
            elif value<row["Norm_Moderate"].values[0] :
                comment = row["Comment_High"].values[0]

        return comment, pass_rules



    def extract(self):
        rules = self.m_rules


        kinematicData = self.m_analysis.kinematicStats

        #phases
        kinematicPhases = self._buildPhases(kinematicData)

        # dataframe iteration
        indexes = rules.index

        out = list()

        print "############################"
        for index in indexes:
            print "----------------------------"
            print index+1
            print "----------------------------"
            row = rules[rules["Id"] == index+1 ] # rules.iloc[i,:]

            if row["Activate"].values[0] == 1:

                if row["Side"].values[0] == "L":
                    contexts = ["Left"]
                elif row["Side"].values[0] == "B":
                    contexts = ["Left","Right"]

                for context in contexts:
                    side = "L" if context == "Left" else "R"
                    variable = str(side +row["Variable"].values[0])


                    if row["Domain"].values[0] == "kinematics":
                        values  = kinematicData.data[variable,context]["mean"][:,row["Plan"].values[0]]
                        val = self._computeMethod(values,context,row["Method"].values[0],row["CyclePeriod"].values[0],kinematicPhases)

                        ruleType = row["Type"].values[0]
                        comment, pass_rules = self._applyRules(val,row,ruleType)

                        if not pass_rules:
                            if row["Alternative_rules"].values[0]>0:
                                alternativeId = row["Alternative_rules"].values[0]

                                rules.loc[rules["Id"] == alternativeId, "Activate"] = 1.0

                                print rules [ rules["Id"] == alternativeId]["Activate"]


                        if comment is not None:

                            print row["Id"].values[0]
                            print row["Rules"].values[0]
                            print context
                            print comment



                            df = pd.DataFrame({'Id' : [row["Id"].values[0]],
                                              'Rules': [row["Rules"].values[0]],
                                              'Context' :[context],
                                              'Comment' :[comment]})
                            out.append(df)
            else:
                print "No comment"

        dataframe = pd.concat(out,ignore_index=True)

        return dataframe
