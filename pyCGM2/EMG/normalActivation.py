# -*- coding: utf-8 -*-

import pyCGM2
from pyCGM2.Utils import files

def getNormalBurstActivity(muscle, fo):
        """
        detection des zones d"activité normal emg
         voici les on pour l"emg
        #            on1=beg
        #            on2=beg+(12/NORMAL_STANCE_PHASE)*(fo-beg)
        #            on3=beg+(48/NORMAL_STANCE_PHASE)*(fo-beg)
        #            on4=fo+((70-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)
        #            on5=fo+((93-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)
        #            on6=fo+((100-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)

        # valeurs normales issues d"une livre EMG analys
        """

        normalActivations = files.openJson(pyCGM2.NORMATIVE_DATABASE_PATH+"emg\\","normalActivation.json")

        NORMAL_STANCE_PHASE=normalActivations["NORMAL_STANCE_PHASE"]
        TABLE = normalActivations["Activation"]

        beg=  0
        end=100

        if muscle in TABLE.keys():
            list_beginBurst=[]
            list_burstDuration=[]

            for i,j in zip(range(0,6,2),range(1,6,2)):
                if TABLE[muscle][i]!="na" and  TABLE[muscle][j]!="na":
                    if TABLE[muscle][i]<NORMAL_STANCE_PHASE:
                        beginBurst=beg+(TABLE[muscle][i]/NORMAL_STANCE_PHASE)*(fo-beg)
                    else:
                        beginBurst=fo+((TABLE[muscle][i]-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)

                    if TABLE[muscle][j]<NORMAL_STANCE_PHASE:
                        endBurst=beg+(TABLE[muscle][j]/NORMAL_STANCE_PHASE)*(fo-beg)
                    else:
                        endBurst=fo+((TABLE[muscle][j]-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)

                    list_beginBurst.append(beginBurst)
                    list_burstDuration.append(endBurst-beginBurst)

        else:
            list_beginBurst=[0]
            list_burstDuration=[0]

        return list_beginBurst,list_burstDuration


def getNormalBurstActivity_fromCycles(muscle,ff,begin, fo, end, apf):
        """
        detection des zones d"activité normal emg
         voici les on pour l"emg
        #            on1=beg
        #            on2=beg+(12/NORMAL_STANCE_PHASE)*(fo-beg)
        #            on3=beg+(48/NORMAL_STANCE_PHASE)*(fo-beg)
        #            on4=fo+((70-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)
        #            on5=fo+((93-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)
        #            on6=fo+((100-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)

        # valeurs normales issues d"une livre EMG analys
        """

        normalActivations = files.openJson(pyCGM2.NORMATIVE_DATABASE_PATH+"emg\\","normalActivation.json")

        NORMAL_STANCE_PHASE=normalActivations["NORMAL_STANCE_PHASE"]
        TABLE = normalActivations["Activation"]


        beg=  (begin-ff)*apf
        fo=(fo-ff)*apf
        end=(end-ff)*apf

        if muscle in TABLE.keys():
            list_beginBurst=[]
            list_burstDuration=[]

            for i,j in zip(range(0,6,2),range(1,6,2)):
                if TABLE[muscle][i]!="na" and  TABLE[muscle][j]!="na":
                    if TABLE[muscle][i]<NORMAL_STANCE_PHASE:
                        beginBurst=beg+(TABLE[muscle][i]/NORMAL_STANCE_PHASE)*(fo-beg)
                    else:
                        beginBurst=fo+((TABLE[muscle][i]-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)

                    if TABLE[muscle][j]<NORMAL_STANCE_PHASE:
                        endBurst=beg+(TABLE[muscle][j]/NORMAL_STANCE_PHASE)*(fo-beg)
                    else:
                        endBurst=fo+((TABLE[muscle][j]-NORMAL_STANCE_PHASE)/(100.0-NORMAL_STANCE_PHASE))*(end-fo)

                    list_beginBurst.append(beginBurst)
                    list_burstDuration.append(endBurst-beginBurst)

        else:
            list_beginBurst=[0]
            list_burstDuration=[0]

        return list_beginBurst,list_burstDuration
