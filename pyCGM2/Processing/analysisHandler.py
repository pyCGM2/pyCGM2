# -*- coding: utf-8 -*-

def getPhases(dataStats):
    #phases
    phases = dict()

    # Left
    phases["Left","ST"] = [0,int(dataStats.pst["stancePhase","Left"]["mean"])]
    phases["Left","SW"] = [int(dataStats.pst["stancePhase","Left"]["mean"]),
                        100]

    phases["Left","DS1"] = [0,
                                    int(dataStats.pst["doubleStance1","Left"]["mean"])]

    phases["Left","SS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"]),
                                    int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"]) ]
    phases["Left","eSS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"]),
                                    int(dataStats.pst["doubleStance1","Left"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Left"]["mean"])]
    phases["Left","mSS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Left"]["mean"]),
                                    int(dataStats.pst["doubleStance1","Left"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Left"]["mean"])]
    phases["Left","lSS"] = [int(dataStats.pst["doubleStance1","Left"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Left"]["mean"]),
                                    int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"])]

    phases["Left","DS2"] = [int(dataStats.pst["stancePhase","Left"]["mean"] - dataStats.pst["doubleStance2","Left"]["mean"]),
                                    int(dataStats.pst["stancePhase","Left"]["mean"])]

    phases["Left","eSW"] = [int(dataStats.pst["stancePhase","Left"]["mean"]),
                                    int(dataStats.pst["stancePhase","Left"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Left"]["mean"])]

    phases["Left","mSW"] = [int(dataStats.pst["stancePhase","Left"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Left"]["mean"]),
                                    int(dataStats.pst["stancePhase","Left"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Left"]["mean"])]

    phases["Left","lSW"] = [int(dataStats.pst["stancePhase","Left"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Left"]["mean"]),
                                    100]

    # Right
    phases["Right","ST"] = [0,int(dataStats.pst["stancePhase","Right"]["mean"])]
    phases["Right","SW"] = [int(dataStats.pst["stancePhase","Right"]["mean"]),
                                    100]

    phases["Right","DS1"] = [0,
                                    int(dataStats.pst["doubleStance1","Right"]["mean"])]
    phases["Right","SS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"]),
                                    int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"]) ]

    phases["Right","eSS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"]),
                                    int(dataStats.pst["doubleStance1","Right"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Right"]["mean"])]
    phases["Right","mSS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"] + 1/3.0* dataStats.pst["simpleStance","Right"]["mean"]),
                                    int(dataStats.pst["doubleStance1","Right"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Right"]["mean"])]
    phases["Right","lSS"] = [int(dataStats.pst["doubleStance1","Right"]["mean"] + 2/3.0* dataStats.pst["simpleStance","Right"]["mean"]),
                                    int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"])]

    phases["Right","DS2"] = [int(dataStats.pst["stancePhase","Right"]["mean"] - dataStats.pst["doubleStance2","Right"]["mean"]),
                                    int(dataStats.pst["stancePhase","Right"]["mean"])]

    phases["Right","eSW"] = [int(dataStats.pst["stancePhase","Right"]["mean"]),
                                    int(dataStats.pst["stancePhase","Right"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Right"]["mean"])]

    phases["Right","mSW"] = [int(dataStats.pst["stancePhase","Right"]["mean"] + 1/3.0* dataStats.pst["swingPhase","Right"]["mean"]),
                                    int(dataStats.pst["stancePhase","Right"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Right"]["mean"])]

    phases["Right","lSW"] = [int(dataStats.pst["stancePhase","Right"]["mean"] + 2/3.0* dataStats.pst["swingPhase","Right"]["mean"]),
                                    100]

    return phases
