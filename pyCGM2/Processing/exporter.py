# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Processing
#APIDOC["Draft"]=False
#--end--

"""
This module contains 2 classes (`XlsAnalysisExportFilter` and `AnalysisExportFilter`)
for exporting an analysis instance in xls fomat and  json respectively.

the class `XlsExportDataFrameFilter` is a generic filter for exporting
Pandas.DataFrame in xls.

"""
import numpy as np
import pandas as pd
import pyCGM2; LOGGER = pyCGM2.LOGGER
from  collections import OrderedDict
import copy


import pyCGM2
from pyCGM2.Utils import files

# ----- PANDAS ---------
# TODO : optimize implementation

FRAMES_HEADER=list()
for i in range(0,101):
    if i>=1 and i<10:
        FRAMES_HEADER.append ( "Frame00"+str(i))
    elif i>9 and i<100:
        FRAMES_HEADER.append ( "Frame0"+str(i))
    else:
        FRAMES_HEADER.append ( "Frame"+str(i))

def isColumnNameExist( dataframe, name):
    if name in dataframe.columns.values:
        LOGGER.logger.debug("column name :[%s] already in the dataFrame",name)

# ---- Spatio temporal parameters -----
def build_df_descriptiveCycle1_1(analysis_outputStats):

    df_collection_L={"mean" : [], "std" : [], "median" : []}
    df_collection_R={"mean" : [], "std" : [], "median" : []}

    for key in analysis_outputStats.keys():

        label = key[0]
        context = key[1]

        if context == "Left":
            for typIt in ["mean","std","median"]:
                df_L = pd.DataFrame({ label : [ analysis_outputStats[label,context][str(typIt)] ],
                                         'EventContext': str(context),
                                         'Stats type' : str(typIt) })
                df_collection_L[typIt].append(df_L)


        if context == "Right":
            for typIt in ["mean","std","median"]:
                df_R = pd.DataFrame({ label : [ analysis_outputStats[label,context][str(typIt)] ],
                                   'EventContext': str(context),
                                   'Stats type': str(typIt)
                                   })
                df_collection_R[typIt].append(df_R)

    left_flag = len(df_collection_L["mean"])
    right_flag = len(df_collection_R["mean"])


    if left_flag:
        df_L_mean=df_collection_L["mean"][0]
        df_L_std=df_collection_L["std"][0]
        df_L_median=df_collection_L["median"][0]
        for i in range(1,len(df_collection_L["mean"])):
            df_L_mean=pd.merge(df_L_mean,df_collection_L["mean"][i],how='outer')
            df_L_std=pd.merge(df_L_std,df_collection_L["std"][i],how='outer')
            df_L_median=pd.merge(df_L_median,df_collection_L["median"][i],how='outer')

    if right_flag:
        df_R_mean=df_collection_R["mean"][0]
        df_R_std=df_collection_R["std"][0]
        df_R_median=df_collection_R["median"][0]
        for i in range(1,len(df_collection_R["mean"])):
            df_R_mean=pd.merge(df_R_mean,df_collection_R["mean"][i],how='outer')
            df_R_std=pd.merge(df_R_std,df_collection_R["std"][i],how='outer')
            df_R_median=pd.merge(df_R_median,df_collection_R["median"][i],how='outer')

    if left_flag and right_flag:
        DF = pd.concat([df_L_mean,df_L_std,df_L_median,
                        df_R_mean,df_R_std,df_R_median],ignore_index=True)
    elif left_flag and not right_flag:
        DF = pd.concat([df_L_mean,df_L_std,df_L_median],ignore_index=True)

    elif not left_flag and  right_flag:
        DF = pd.concat([df_R_mean,df_R_std,df_R_median],ignore_index=True)

    return DF


def build_df_cycles1_1(analysis_outputStats):

    df_collection_L=[]
    df_collection_R=[]

    for key in analysis_outputStats.keys():
        label = key[0]
        context = key[1]
        n =len(analysis_outputStats[label,context]['values'])

        if context == "Left":
            df_L = pd.DataFrame({ label : analysis_outputStats[label,context]['values'],
                               "Cycle": range(0,n),
                               'EventContext': str(context)
                               })

            df_collection_L.append(df_L)


        if context == "Right":
            df_R = pd.DataFrame({ label : analysis_outputStats[label,context]['values'],
                               "Cycle": range(0,n),
                               'EventContext': str(context)
                               })

            df_collection_R.append(df_R)

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)

    if left_flag:
        df_L=df_collection_L[0]
        for i in range(1,len(df_collection_L)):
            df_L=pd.merge(df_L,df_collection_L[i])

    if right_flag:
        df_R=df_collection_R[0]
        for i in range(1,len(df_collection_R)):
            df_R=pd.merge(df_R,df_collection_R[i])

    if left_flag and right_flag:
        DF = pd.concat([df_L,df_R],ignore_index=True)
    elif left_flag and not right_flag:
        DF = df_L
    elif not left_flag and  right_flag:
       DF = df_R

    return DF


# ---- for Scores -----
def build_df_descriptiveCycle1_1_overall(analysis_outputStats,columnName):

    df_collection=[]

    valuesMean=analysis_outputStats["mean"]
    df=pd.DataFrame(valuesMean,  columns = [columnName])
    df['EventContext']= "overall"
    df['Stats type'] = "mean"
    df_collection.append(df)

    valuesStd=analysis_outputStats["std"]
    df=pd.DataFrame(valuesStd,  columns = [columnName])
    df['EventContext']= "overall"
    df['Stats type'] = "std"
    df_collection.append(df)

    valuesMedian=analysis_outputStats["median"]
    df=pd.DataFrame(valuesMedian,  columns = [columnName])
    df['EventContext']= "overall"
    df['Stats type'] = "median"
    df_collection.append(df)

    df=pd.concat(df_collection,ignore_index=True)

    return df



def build_df_descriptiveCycle1_1_onlyContext(analysis_outputStats,columnName):


    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for context in analysis_outputStats.keys():

        if context == "Left":
            valuesMean=analysis_outputStats[context]["mean"]
            df=pd.DataFrame(valuesMean,  columns = [columnName])
            df['EventContext']= context
            df['Stats type'] = "mean"
            df_collection_L.append(df)

            valuesStd=analysis_outputStats[context]["std"]
            df=pd.DataFrame(valuesStd,  columns= [columnName])
            df['EventContext']= context
            df['Stats type'] = "std"
            df_collection_L.append(df)

            valuesMedian=analysis_outputStats[context]["median"]
            df=pd.DataFrame(valuesMedian,  columns= [columnName])
            df['EventContext']= context
            df['Stats type'] = "median"
            df_collection_L.append(df)


        if context == "Right":
            valuesMean=analysis_outputStats[context]["mean"]
            df=pd.DataFrame(valuesMean,  columns = [columnName])
            df['EventContext']= context
            df['Stats type'] = "mean"
            df_collection_R.append(df)

            valuesStd=analysis_outputStats[context]["std"]
            df=pd.DataFrame(valuesStd,  columns= [columnName])
            df['EventContext']= context
            df['Stats type'] = "std"
            df_collection_R.append(df)

            valuesMedian=analysis_outputStats[context]["median"]
            df=pd.DataFrame(valuesMedian,  columns= [columnName])
            df['EventContext']= context
            df['Stats type'] = "median"
            df_collection_R.append(df)

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)

    if left_flag:  df_L=pd.concat(df_collection_L,ignore_index=True)
    if right_flag: df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        DF = pd.concat([df_L,df_R],ignore_index=True)
    if left_flag and not right_flag:
        DF = df_L
    if not left_flag and right_flag:
        DF = df_R

    return DF


def build_df_cycles1_1_onlyContext(analysis_outputStats, columnName):

    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for context in analysis_outputStats.keys():
        if context =="Left" : # FIXME TODO : if different form L and R
            values = analysis_outputStats[context]["values"]
            df=pd.DataFrame(values,  columns= [columnName])
            df['Cycle']= 0 if  len(values) == 1 else range(0,len(values))
            df['EventContext']= context
            df_collection_L.append(df)

        if context=="Right" :
            values = analysis_outputStats[context]["values"]
            df=pd.DataFrame(values,  columns= [columnName])
            df['Cycle']= 0 if  len(values) == 1 else range(0,len(values))
            df['EventContext']= context
            df_collection_R.append(df)

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)

    if left_flag:
        df_L=pd.concat(df_collection_L,ignore_index=True)

    if right_flag:
        df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        df_cycles = pd.concat([df_L,df_R],ignore_index=True)

    if left_flag and not right_flag:
        df_cycles = df_L

    if not left_flag and right_flag:
        df_cycles = df_R

    DF = df_cycles


    return DF




def buid_df_descriptiveCycle1_3(analysis_outputStats,columnName):


    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for key in analysis_outputStats.keys():
        label = key[0]
        context = key[1]
        if context == "Left":
            valuesMean=analysis_outputStats[label,context]["mean"]
            df=pd.DataFrame(valuesMean,  columns = [columnName])
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['EventContext']= context
            df['Stats type'] = "mean"
            df_collection_L.append(df)

            valuesStd=analysis_outputStats[label,context]["std"]
            df=pd.DataFrame(valuesStd,  columns= [columnName])
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['EventContext']= context
            df['Stats type'] = "std"
            df_collection_L.append(df)

            valuesMedian=analysis_outputStats[label,context]["median"]
            df=pd.DataFrame(valuesMedian,  columns= [columnName])
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['EventContext']= context
            df['Stats type'] = "median"
            df_collection_L.append(df)


        if context == "Right":
            valuesMean=analysis_outputStats[label,context]["mean"]
            df=pd.DataFrame(valuesMean,  columns= [columnName])
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['EventContext']= context
            df['Stats type'] = "mean"
            df_collection_R.append(df)

            valuesStd=analysis_outputStats[label,context]["std"]
            df=pd.DataFrame(valuesStd,  columns= [columnName])
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['EventContext']= context
            df['Stats type'] = "std"
            df_collection_R.append(df)

            valuesMedian=analysis_outputStats[label,context]["median"]
            df=pd.DataFrame(valuesMedian,  columns= [columnName])
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['EventContext']= context
            df['Stats type'] = "median"
            df_collection_R.append(df)

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)

    if left_flag:  df_L=pd.concat(df_collection_L,ignore_index=True)
    if right_flag: df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        DF = pd.concat([df_L,df_R],ignore_index=True)
    if left_flag and not right_flag:
        DF = df_L
    if not left_flag and right_flag:
        DF = df_R

    return DF


def build_df_cycles1_3(analysis_outputStats, columnName):

    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for key in analysis_outputStats.keys():
        label = key[0]
        context = key[1]
        i_l=0
        i_r=0
        if context =="Left" : # FIXME TODO : if different form L and R
            for itValues in analysis_outputStats[label,context]["values"]:
                df=pd.DataFrame(itValues,  columns= [columnName])
                df['Axis']=['X','Y','Z']
                df['Label']=label
                df['Cycle']= i_l
                df['EventContext']= context
                df_collection_L.append(df)
                i_l+=1

        if context=="Right" :
            for itValues in analysis_outputStats[label,context]["values"]:
                df=pd.DataFrame(itValues,  columns= [columnName])
                df['Axis']=['X','Y','Z']
                df['Label']=label
                df['Cycle']= i_r
                df['EventContext']= context
                df_collection_R.append(df)
                i_r+=1

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)

    if left_flag:
        df_L=pd.concat(df_collection_L,ignore_index=True)

    if right_flag:
        df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        df_cycles = pd.concat([df_L,df_R],ignore_index=True)

    if left_flag and not right_flag:
        df_cycles = df_L

    if not left_flag and right_flag:
        df_cycles = df_R

    DF = df_cycles


    return DF


# ---- for 101-frames data -----
def build_df_descriptiveCycle101_3(analysis_outputStats):


    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for key in analysis_outputStats.data.keys():
        label = key[0]
        context = key[1]
        if context == "Left":

            valuesMean=analysis_outputStats.data[label,context]["mean"]
            if not np.all(valuesMean==0):
                df=pd.DataFrame(valuesMean.T,  columns = FRAMES_HEADER)
                df['Axe']=['X','Y','Z']
                df['Label']=label
                df['EventContext']= context
                df['Stats type'] = "mean"
                df_collection_L.append(df)

            valuesStd=analysis_outputStats.data[label,context]["std"]
            if not np.all(valuesStd==0):
                df=pd.DataFrame(valuesStd.T,  columns= FRAMES_HEADER)
                df['Axe']=['X','Y','Z']
                df['Label']=label
                df['EventContext']= context
                df['Stats type'] = "std"
                df_collection_L.append(df)

            valuesMedian=analysis_outputStats.data[label,context]["median"]
            if not np.all(valuesMedian==0):
                df=pd.DataFrame(valuesMedian.T,  columns= FRAMES_HEADER)
                df['Axe']=['X','Y','Z']
                df['Label']=label
                df['EventContext']= context
                df['Stats type'] = "median"
                df_collection_L.append(df)


        if context == "Right":

            valuesMean=analysis_outputStats.data[label,context]["mean"]
            if not np.all(valuesMean==0):
                df=pd.DataFrame(valuesMean.T,  columns= FRAMES_HEADER)
                df['Axe']=['X','Y','Z']
                df['Label']=label
                df['EventContext']= context
                df['Stats type'] = "mean"
                df_collection_R.append(df)

            valuesStd=analysis_outputStats.data[label,context]["std"]
            if not np.all(valuesStd==0):
                df=pd.DataFrame(valuesStd.T,  columns= FRAMES_HEADER)
                df['Axe']=['X','Y','Z']
                df['Label']=label
                df['EventContext']= context
                df['Stats type'] = "std"
                df_collection_R.append(df)

            valuesMedian=analysis_outputStats.data[label,context]["median"]
            if not np.all(valuesMedian==0):
                df=pd.DataFrame(valuesMedian.T,  columns= FRAMES_HEADER)
                df['Axe']=['X','Y','Z']
                df['Label']=label
                df['EventContext']= context
                df['Stats type'] = "median"
                df_collection_R.append(df)

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)

    if left_flag:  df_L=pd.concat(df_collection_L,ignore_index=True)
    if right_flag: df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        DF = pd.concat([df_L,df_R],ignore_index=True)
    if left_flag and not right_flag:
        DF = df_L
    if not left_flag and right_flag:
        DF = df_R

    return DF




def build_df_cycles101_3(analysis_outputStats):

    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for key in analysis_outputStats.data.keys():
        label = key[0]
        context = key[1]
        i_l=0
        i_r=0
        if context =="Left" :
            for itValues in analysis_outputStats.data[label,context]["values"]:
                if not np.all(itValues==0):
                    df=pd.DataFrame(itValues.T,  columns= FRAMES_HEADER)
                    df['Axis']=['X','Y','Z']
                    df['Label']=label
                    df['Cycle']= i_l
                    df['EventContext']= context
                    df_collection_L.append(df)
                i_l+=1 # will serve for merging with spt

        if context=="Right" :
            for itValues in analysis_outputStats.data[label,context]["values"]:
                if not np.all(itValues==0):
                    df=pd.DataFrame(itValues.T,  columns= FRAMES_HEADER)
                    df['Axis']=['X','Y','Z']
                    df['Label']=label
                    df['Cycle']= i_r
                    df['EventContext']= context
                    df_collection_R.append(df)
                i_r+=1 # will serve for merging with spt

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)



    if left_flag:
        df_L=pd.concat(df_collection_L,ignore_index=True)

    if right_flag:
        df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        df_cycles = pd.concat([df_L,df_R],ignore_index=True)

    if left_flag and not right_flag:
        df_cycles = df_L

    if not left_flag and right_flag:
        df_cycles = df_R


    # pst section
    if analysis_outputStats.pst !={}:
        df_collection_pst_L=[]
        df_collection_pst_R=[]

        for key in analysis_outputStats.pst.keys():
            label = key[0]
            context = key[1]
            n =len(analysis_outputStats.pst[label,context]['values'])

            if context == "Left":
                df_pst_L = pd.DataFrame({ label : analysis_outputStats.pst[label,context]['values'],
                                   "Cycle": range(0,n),
                                   'EventContext': str(context)
                                   })

                df_collection_pst_L.append(df_pst_L)

            if context == "Right":
                df_pst_R = pd.DataFrame({ label : analysis_outputStats.pst[label,context]['values'],
                                   "Cycle": range(0,n),
                                   'EventContext': str(context)
                                   })

                df_collection_pst_R.append(df_pst_R)

        if left_flag:
            df_pst_L=df_collection_pst_L[0]
            for i in range(1,len(df_collection_pst_L)):
                df_pst_L=pd.merge(df_pst_L,df_collection_pst_L[i])

        if right_flag:
            df_pst_R=df_collection_pst_R[0]
            for i in range(1,len(df_collection_pst_R)):
                df_pst_R=pd.merge(df_pst_R,df_collection_pst_R[i])

        if left_flag and right_flag:
            df_pst = pd.concat([df_pst_L,df_pst_R],ignore_index=True)
        if left_flag and not right_flag:
            df_pst = df_pst_L
        if not left_flag and right_flag:
            df_pst = df_pst_R


        # merging
        DF=pd.merge(df_cycles,df_pst, on=['Cycle','EventContext'],how='outer')

    else:
        DF = df_cycles


    return DF

def build_df_descriptiveCycle101_1(analysis_outputStats):


    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for key in analysis_outputStats.data.keys():
        label = key[0]
        context = key[1]
        if context == "Left":

            valuesMean=analysis_outputStats.data[label,context]["mean"]

            if not np.all(np.isnan(valuesMean)):
                if not np.all(valuesMean==0):
                    df=pd.DataFrame(valuesMean.T,  columns = FRAMES_HEADER)
                    df['Label']=label
                    df['EventContext']= context
                    df['Stats type'] = "mean"
                    df_collection_L.append(df)

                valuesStd=analysis_outputStats.data[label,context]["std"]
                if not np.all(valuesStd==0):
                    df=pd.DataFrame(valuesStd.T,  columns= FRAMES_HEADER)
                    df['Label']=label
                    df['EventContext']= context
                    df['Stats type'] = "std"
                    df_collection_L.append(df)

                valuesMedian=analysis_outputStats.data[label,context]["median"]
                if not np.all(valuesMedian==0):
                    df=pd.DataFrame(valuesMedian.T,  columns= FRAMES_HEADER)
                    df['Label']=label
                    df['EventContext']= context
                    df['Stats type'] = "median"
                    df_collection_L.append(df)


        if context == "Right":
            valuesMean=analysis_outputStats.data[label,context]["mean"]
            if not np.all(np.isnan(valuesMean)):
                if not np.all(valuesMean==0):
                    df=pd.DataFrame(valuesMean.T,  columns= FRAMES_HEADER)
                    df['Label']=label
                    df['EventContext']= context
                    df['Stats type'] = "mean"
                    df_collection_R.append(df)

                valuesStd=analysis_outputStats.data[label,context]["std"]
                if not np.all(valuesStd==0):
                    df=pd.DataFrame(valuesStd.T,  columns= FRAMES_HEADER)
                    df['Label']=label
                    df['EventContext']= context
                    df['Stats type'] = "std"
                    df_collection_R.append(df)

                valuesMedian=analysis_outputStats.data[label,context]["median"]
                if not np.all(valuesMedian==0):
                    df=pd.DataFrame(valuesMedian.T,  columns= FRAMES_HEADER)
                    df['Label']=label
                    df['EventContext']= context
                    df['Stats type'] = "median"
                    df_collection_R.append(df)

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)

    if left_flag:  df_L=pd.concat(df_collection_L,ignore_index=True)
    if right_flag: df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        DF = pd.concat([df_L,df_R],ignore_index=True)
    if left_flag and not right_flag:
        DF = df_L
    if not left_flag and right_flag:
        DF = df_R

    return DF


def build_df_cycles101_1(analysis_outputStats):

    # "data" section
    df_collection_L=[]
    df_collection_R=[]

    for key in analysis_outputStats.data.keys():
        label = key[0]
        context = key[1]
        i_l=0
        i_r=0
        if context =="Left" :
            for itValues in analysis_outputStats.data[label,context]["values"]:
                if not np.all(itValues==0):
                    df=pd.DataFrame(itValues.T,  columns= FRAMES_HEADER)
                    df['Label']=label
                    df['Cycle']= i_l
                    df['EventContext']= context
                    df_collection_L.append(df)
                i_l+=1 # will serve for merging with spt

        if context=="Right" :
            for itValues in analysis_outputStats.data[label,context]["values"]:
                if not np.all(itValues==0):
                    df=pd.DataFrame(itValues.T,  columns= FRAMES_HEADER)
                    df['Label']=label
                    df['Cycle']= i_r
                    df['EventContext']= context
                    df_collection_R.append(df)
                i_r+=1 # will serve for merging with spt

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)



    if left_flag:
        df_L=pd.concat(df_collection_L,ignore_index=True)

    if right_flag:
        df_R=pd.concat(df_collection_R,ignore_index=True)

    if left_flag and right_flag:
        df_cycles = pd.concat([df_L,df_R],ignore_index=True)

    if left_flag and not right_flag:
        df_cycles = df_L

    if not left_flag and right_flag:
        df_cycles = df_R


    # pst section
    if analysis_outputStats.pst !={}:
        df_collection_pst_L=[]
        df_collection_pst_R=[]

        for key in analysis_outputStats.pst.keys():
            label = key[0]
            context = key[1]
            n =len(analysis_outputStats.pst[label,context]['values'])

            if context == "Left":
                df_pst_L = pd.DataFrame({ label : analysis_outputStats.pst[label,context]['values'],
                                   "Cycle": range(0,n),
                                   'EventContext': str(context)
                                   })

                df_collection_pst_L.append(df_pst_L)

            if context == "Right":
                df_pst_R = pd.DataFrame({ label : analysis_outputStats.pst[label,context]['values'],
                                   "Cycle": range(0,n),
                                   'EventContext': str(context)
                                   })

                df_collection_pst_R.append(df_pst_R)

        if left_flag:
            df_pst_L=df_collection_pst_L[0]
            for i in range(1,len(df_collection_pst_L)):
                df_pst_L=pd.merge(df_pst_L,df_collection_pst_L[i])

        if right_flag:
            df_pst_R=df_collection_pst_R[0]
            for i in range(1,len(df_collection_pst_R)):
                df_pst_R=pd.merge(df_pst_R,df_collection_pst_R[i])

        if left_flag and right_flag:
            df_pst = pd.concat([df_pst_L,df_pst_R],ignore_index=True)
        if left_flag and not right_flag:
            df_pst = df_pst_L
        if not left_flag and right_flag:
            df_pst = df_pst_R


        # merging
        DF=pd.merge(df_cycles,df_pst, on=['Cycle','EventContext'],how='outer')

    else:
        DF = df_cycles


    return DF


def renameEmgInAnalysis(analysisInstance,emgSettings):

    emgChannels = list()
    emgContexts = list()
    emgMuscles = list()
    for channel in emgSettings["CHANNELS"].keys():
        if emgSettings["CHANNELS"][channel]["Muscle"] is not None and  emgSettings["CHANNELS"][channel]["Muscle"] != "None":
            emgChannels.append(channel)
            emgContexts.append(emgSettings["CHANNELS"][channel]["Context"])
            emgMuscles.append(emgSettings["CHANNELS"][channel]["Muscle"])

    copiedAnalysis = copy.deepcopy(analysisInstance)

    newLabel = dict()
    for i in range(0,len(emgChannels)):
        newLabel[emgChannels[i]] = emgContexts[i][0] + emgMuscles[i]

    for keyIt in copiedAnalysis.emgStats.data.keys():
        label = keyIt[0]
        context = keyIt[1]
        channel = label[0:label.find("_")]
        if channel  in  newLabel.keys():
            del analysisInstance.emgStats.data[keyIt[0],context]

            newLabelFinal = keyIt[0].replace(channel,newLabel[channel])
            content = copiedAnalysis.emgStats.data[keyIt[0],context]

            analysisInstance.emgStats.data[newLabelFinal,context] = content
            LOGGER.logger.debug("label [%s] replaced with [%s]"%(keyIt[0],newLabelFinal))


class XlsExportDataFrameFilter(object):
    """Filter exporting a pandas.dataFrame or a list of pandas.dataFrame as xls spreadsheet(s)
    """

    def __init__(self):

        self.dataframes =list()

    def setDataFrames(self, dataframes):
        """Set the dataFrame

        Args:
            dataframes (pandas.dataFrame or list): dataframe or list of dataframe

        """

        if isinstance(dataframes,pd.core.frame.DataFrame):
            dataframes=[dataframes]

        for it in dataframes:
            self.dataframes.append(it)

    def export(self,outputName, path=None,excelFormat = "xls"):
        """ Export spreadsheet

        Args:
            outputName (str): filename without extension
            path (str): Path
            excelFormat (str): format (xls,xlsx)

        """
        i=0
        for  dataframe in self.dataframes:
            if path == None:
                if excelFormat == "xls":
                    xlsxWriter = pd.ExcelWriter((outputName + "- dataframe.xls"),engine='xlwt')
                elif excelFormat == "xlsx":
                    xlsxWriter = pd.ExcelWriter((outputName + "- dataframe.xlsx"))
            else:
                if excelFormat == "xls":
                    xlsxWriter = pd.ExcelWriter((path+outputName + "- dataframe.xls"),engine='xlwt')
                elif excelFormat == "xlsx":
                    xlsxWriter = pd.ExcelWriter((path+outputName + "- dataFrame.xlsx"))

            dataframe.to_excel(xlsxWriter,"dataframe_"+str(i),index=False)
            i+=1

        xlsxWriter.save()

class XlsAnalysisExportFilter(object):
    """Filter exporting an `analysis` instance as spreadheet.

    Exported spreadsheet can organize either by row (`Advanced` mode) or column
    (`basic` mode). In `Advanced` mode, a row is made up of 101 columns representing
    a time normalized cycle.


    """

    def __init__(self):

        self.analysis = None

    def setAnalysisInstance(self,analysisInstance):
        """set the `analysis` instance

        Args:
            analysisInstance (pyCGM2.analysis.Analysis): an `analysis` instance
        """

        self.analysis = analysisInstance

    def export(self,outputName, path=None,excelFormat = "xls",mode="Advanced"):
        """ Export spreadsheet

        Args:
            outputName (str): filename without extension
            path (str): Path
            excelFormat (str): format (xls,xlsx)
            mode (str): structure mode of the spreadsheet (Advanced,Basic)

        """


        if mode == "Advanced":
            self.__advancedExport(outputName, path=path, excelFormat = excelFormat)
        elif mode == "Basic":
            self.__basicExport(outputName, path=path, excelFormat = excelFormat)

    def __basicExport(self,outputName, path=None,excelFormat = "xls"):
        if path == None:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter((outputName + "- basic.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter((outputName + "- basic.xlsx"))
        else:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter((path+outputName + "- basic.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter((path+outputName + "- basic.xlsx"))

        # metadata
        #--------------
        if self.analysis.subjectInfo is not  None:
            subjInfo =  self.analysis.subjectInfo
        else:
            subjInfo=None

        if self.analysis.modelInfo is not  None:
            modelInfo =  self.analysis.modelInfo
        else:
            modelInfo=None

        if self.analysis.experimentalInfo is not  None:
            experimentalConditionInfo =  self.analysis.experimentalInfo
        else:
            experimentalConditionInfo=None

        list_index =list()# use for sorting index
        if subjInfo is not None:
            for key in subjInfo:
                list_index.append(key)
            serie_subject = pd.Series(subjInfo)
        else:
            serie_subject = pd.Series()

        if modelInfo is not None:
            for key in modelInfo:
                list_index.append(key)
            serie_model = pd.Series(modelInfo)
        else:
            serie_model = pd.Series()

        if experimentalConditionInfo is not None:
            for key in experimentalConditionInfo:
                list_index.append(key)
            serie_exp = pd.Series(experimentalConditionInfo)
        else:
            serie_exp = pd.Series()

        df_metadata = pd.DataFrame({"subjectInfos": serie_subject,
                                    "modelInfos": serie_model,
                                    "experimentInfos" : serie_exp},
                                    index = list_index)


        df_metadata.to_excel(xlsxWriter,"Infos",index=False)

        if self.analysis.kinematicStats.data!={}:

            # spatio temporal paramaters matching Kinematic cycles
            dfs_l =[]
            dfs_r =[]
            for key in self.analysis.kinematicStats.pst.keys():
                label = key[0]
                context = key[1]
                n =len(self.analysis.kinematicStats.pst[label,context]['values'])
                cycle_header= ["Cycle "+str(i) for i in range(0,n)]

                if context == "Left":
                    df = pd.DataFrame.from_items([(label, self.analysis.kinematicStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_l.append(df)

                if context == "Right":
                    df = pd.DataFrame.from_items([(label, self.analysis.kinematicStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_r.append(df)

            if dfs_l !=[]:
                df_pst_L=pd.concat(dfs_l)
                df_pst_L.to_excel(xlsxWriter,"Left - stp-kinematics",index=False)

            if dfs_r !=[]:
                df_pst_R=pd.concat(dfs_r)
                df_pst_R.to_excel(xlsxWriter,"Right - stp-kinematics",index=False)


            # kinematic cycles
            for key in self.analysis.kinematicStats.data.keys():
                label = key[0]
                context = key[1]

                X=[]
                Y=[]
                Z=[]

                countCycle = 0
                for cycleValuesIt  in self.analysis.kinematicStats.data[label,context]["values"]:
                    if not np.all(cycleValuesIt == 0):
                        X.append(cycleValuesIt[:,0])
                        Y.append(cycleValuesIt[:,0])
                        Z.append(cycleValuesIt[:,0])
                        countCycle+=1

                X = np.asarray(X).T
                Y = np.asarray(Y).T
                Z = np.asarray(Z).T

                if X.size!=0 and Y.size!=0 and Z.size!=0:

                    cycle_header= ["Cycle "+str(i) for i in range(0,countCycle)]
                    frame_header= ["Frame "+str(i) for i in range(0,101)]


                    df_x=pd.DataFrame(X,  columns= cycle_header,index = frame_header )
                    df_x['Axis']='X'
                    df_y=pd.DataFrame(Y,  columns= cycle_header,index = frame_header )
                    df_y['Axis']='Y'
                    df_z=pd.DataFrame(Z,  columns= cycle_header,index = frame_header )
                    df_z['Axis']='Z'

                    df_label = pd.concat([df_x,df_y,df_z])
                    df_label.to_excel(xlsxWriter,str(label+"."+context),index=False)

        if self.analysis.kineticStats.data!={}:
            # spatio temporal paramaters matching Kinetic cycles
            dfs_l =[]
            dfs_r =[]
            for key in self.analysis.kineticStats.pst.keys():
                label = key[0]
                context = key[1]
                n =len(self.analysis.kineticStats.pst[label,context]['values'])
                cycle_header= ["Cycle "+str(i) for i in range(0,n)]

                if context == "Left":
                    df = pd.DataFrame.from_items([(label, self.analysis.kineticStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_l.append(df)

                if context == "Right":
                    df = pd.DataFrame.from_items([(label, self.analysis.kineticStats.pst[label,context]['values'])],orient='index', columns=cycle_header)
                    dfs_r.append(df)

            if dfs_l !=[]:
                df_pst_L=pd.concat(dfs_l)
                df_pst_L.to_excel(xlsxWriter,"Left - pst-kinetics",index=False)

            if dfs_r !=[]:
                df_pst_R=pd.concat(dfs_r)
                df_pst_R.to_excel(xlsxWriter,"Right - pst-kinetics",index=False)


            # kinetic cycles
            for key in self.analysis.kineticStats.data.keys():
                label=key[0]
                context=key[1]

                X=[]
                Y=[]
                Z=[]

                countCycle = 0
                for cycleValuesIt  in self.analysis.kineticStats.data[label,context]["values"]:
                    if not np.all(cycleValuesIt == 0):
                        X.append(cycleValuesIt[:,0])
                        Y.append(cycleValuesIt[:,0])
                        Z.append(cycleValuesIt[:,0])
                        countCycle+=1

                X = np.asarray(X).T
                Y = np.asarray(Y).T
                Z = np.asarray(Z).T


                if X.size!=0 and Y.size!=0 and Z.size!=0:
                    cycle_header= ["Cycle "+str(i) for i in range(0,n)]
                    frame_header= ["Frame "+str(i) for i in range(0,101)]

                    df_x=pd.DataFrame(X,  columns= cycle_header,index = frame_header )
                    df_x['Axis']='X'
                    df_y=pd.DataFrame(Y,  columns= cycle_header,index = frame_header )
                    df_y['Axis']='Y'
                    df_z=pd.DataFrame(Z,  columns= cycle_header,index = frame_header )
                    df_z['Axis']='Z'

                    df_label = pd.concat([df_x,df_y,df_z])
                    df_label.to_excel(xlsxWriter,str(label+"."+context),index=False)

        xlsxWriter.save()
        LOGGER.logger.info("basic dataFrame [%s- basic] Exported"%outputName)





    def __advancedExport(self,outputName, path=None, excelFormat = "xls",csvFileExport =False):
        if path == None:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter((outputName + "- Advanced.xls"),engine='xlwt',encoding='utf-8')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter((outputName + "- Advanced.xlsx"),encoding='utf-8')
        else:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter((path+outputName + "- Advanced.xls"),engine='xlwt',encoding='utf-8')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter((path+outputName + "- Advanced.xlsx"),encoding='utf-8')

        # infos
        #-------
        if self.analysis.modelInfo is not None:
            modelInfo =  self.analysis.modelInfo
        else:
            modelInfo=None


        if self.analysis.subjectInfo is not None:
            subjInfo =  self.analysis.subjectInfo
        else:
            subjInfo=None

        if self.analysis.experimentalInfo is not None:
            condExpInfo =  self.analysis.experimentalInfo
        else:
            condExpInfo=None

        # spatio temporal parameters
        #---------------------------

        if self.analysis.stpStats != {} and self.analysis.stpStats is not None :

            # stage 1 : get descriptive data
            # --------------------------------
            df_descriptiveStp = build_df_descriptiveCycle1_1(self.analysis.stpStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value

            if self.analysis.stpInfo is not None:
                for key,value in self.analysis.stpInfo.items():
                    isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value

            df_descriptiveStp.to_excel(xlsxWriter,'descriptive stp',index=False)


            # stage 2 : get cycle values
            # --------------------------------
            df_stp = build_df_cycles1_1(self.analysis.stpStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_stp, key)
                    df_stp[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_stp, key)
                    df_stp[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_stp, key)
                    df_stp[key] = value

            if self.analysis.stpInfo is not None:
                for key,value in self.analysis.stpInfo.items():
                    isColumnNameExist( df_stp, key)
                    df_stp[key] = value

            df_stp.to_excel(xlsxWriter,'stp cycles',index=False)

            if csvFileExport:
                if path == None:
                    df_stp.to_csv((outputName + " - stp - DataFrame.csv"),sep=";")
                else:
                    df_stp.to_csv((path+outputName + " - stp - DataFrame.csv"),sep=";")

        # Scores
        #---------------------------

        # GPS
        if self.analysis.gps is not None:

            build_df_cycles1_1_onlyContext(self.analysis.gps["Context"], "Gps")


            df_descriptiveGpsByContext = build_df_descriptiveCycle1_1_onlyContext(self.analysis.gps["Context"], "Gps")
            df_descriptiveGpsOverall = build_df_descriptiveCycle1_1_overall(self.analysis.gps["Overall"],"Gps")
            df_descriptiveGps =  pd.concat([df_descriptiveGpsOverall,df_descriptiveGpsByContext])

            df_allGpsByContext = build_df_cycles1_1_onlyContext(self.analysis.gps["Context"], "Gps")

            # add infos
            for itdf in [df_descriptiveGps,df_descriptiveGpsByContext,df_allGpsByContext]:
                if modelInfo is not None:
                    for key,value in modelInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value

                if subjInfo is not None:
                    for key,value in subjInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value
                if condExpInfo is not None:
                    for key,value in condExpInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value

                if self.analysis.scoreInfo is not None:
                    for key,value in self.analysis.scoreInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value

            df_descriptiveGps.to_excel(xlsxWriter,'descriptive GPS',index=False)
            df_allGpsByContext.to_excel(xlsxWriter,'GPS cycles',index=False)


        if self.analysis.gvs is not None:
            df_descriptiveGvs = buid_df_descriptiveCycle1_3(self.analysis.gvs,"Gvs")
            df_allGvs = build_df_cycles1_3(self.analysis.gvs, "Gvs")

            # add infos
            for itdf in [df_descriptiveGvs,df_allGvs]:
                if modelInfo is not None:
                    for key,value in modelInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value

                if subjInfo is not None:
                    for key,value in subjInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value
                if condExpInfo is not None:
                    for key,value in condExpInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value

                if self.analysis.scoreInfo is not None:
                    for key,value in self.analysis.scoreInfo.items():
                        isColumnNameExist( itdf, key)
                        itdf[key] = value


            df_descriptiveGvs.to_excel(xlsxWriter,'descriptive GVS',index=False)
            df_allGvs.to_excel(xlsxWriter,'GVS cycles',index=False)


        # Kinematics ouput
        #---------------------------


        if self.analysis.kinematicStats.data!={}:

            # stage 1 : get descriptive data
            # --------------------------------
            df_descriptiveKinematics = build_df_descriptiveCycle101_3(self.analysis.kinematicStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value

            if self.analysis.kinematicInfo is not None:
                for key,value in self.analysis.kinematicInfo.items():
                    isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value

            df_descriptiveKinematics.to_excel(xlsxWriter,'descriptive kinematics',index=False)

            # stage 2 : get cycle values
            # --------------------------------

            # cycles
            df_kinematics =  build_df_cycles101_3(self.analysis.kinematicStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value

            if self.analysis.kinematicInfo is not None:
                for key,value in self.analysis.kinematicInfo.items():
                    isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value

            df_kinematics.to_excel(xlsxWriter,'Kinematic cycles',index=False)
            if csvFileExport:
                if path == None:
                    df_kinematics.to_csv((outputName + " - kinematics - DataFrame.csv"),sep=";")
                else:
                    df_kinematics.to_csv((path+outputName + " - kinematics - DataFrame.csv"),sep=";")


        # Kinetic ouputs
        #---------------------------
        if self.analysis.kineticStats.data!={}:

            # stage 1 : get descriptive data
            # --------------------------------
            df_descriptiveKinetics = build_df_descriptiveCycle101_3(self.analysis.kineticStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value

            if self.analysis.kineticInfo is not None:
                for key,value in self.analysis.kineticInfo.items():
                    isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value

            df_descriptiveKinetics.to_excel(xlsxWriter,'descriptive kinetics',index=False)

            # stage 2 : get cycle values
            # --------------------------------

            # cycles
            df_kinetics =  build_df_cycles101_3(self.analysis.kineticStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value

            if self.analysis.kineticInfo is not None:
                for key,value in self.analysis.kineticInfo.items():
                    isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value

            df_kinetics.to_excel(xlsxWriter,'Kinetic cycles',index=False)
            if csvFileExport:
                if path == None:
                    df_kinetics.to_csv((outputName + " - kinetics - DataFrame.csv"),sep=";")
                else:
                    df_kinetics.to_csv((path+outputName + " - kinetics - DataFrame.csv"),sep=";")


        # EMG ouputs
        #---------------------------
        if self.analysis.emgStats.data!={}:

            # stage 1 : get descriptive data
            # --------------------------------
            df_descriptiveEMG = build_df_descriptiveCycle101_1(self.analysis.emgStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_descriptiveEMG, key)
                    df_descriptiveEMG[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_descriptiveEMG, key)
                    df_descriptiveEMG[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_descriptiveEMG, key)
                    df_descriptiveEMG[key] = value

            if self.analysis.emgInfo is not None:
                for key,value in self.analysis.emgInfo.items():
                    isColumnNameExist( df_descriptiveEMG, key)
                    df_descriptiveEMG[key] = value

            df_descriptiveEMG.to_excel(xlsxWriter,'descriptive EMG',index=False)

            # stage 2 : get cycle values
            # --------------------------------

            # cycles
            df_emg =  build_df_cycles101_1(self.analysis.emgStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    isColumnNameExist( df_emg, key)
                    df_emg[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    isColumnNameExist( df_emg, key)
                    df_emg[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    isColumnNameExist( df_emg, key)
                    df_emg[key] = value

            if self.analysis.emgInfo is not None:
                for key,value in self.analysis.emgInfo.items():
                    isColumnNameExist( df_emg, key)
                    df_emg[key] = value

            df_emg.to_excel(xlsxWriter,'EMG cycles',index=False)
            if csvFileExport:
                if path == None:
                    df_emg.to_csv((outputName + " - EMG - DataFrame.csv"),sep=";")
                else:
                    df_emg.to_csv((path+outputName + " - EMG - DataFrame.csv"),sep=";")


        LOGGER.logger.info("advanced dataFrame [%s- Advanced] Exported"%outputName)

        xlsxWriter.save()


class AnalysisExportFilter(object):
    """Filter exporting an `analysis` instance as json.
    """

    def __init__(self):

        self.analysis = None

    def setAnalysisInstance(self,analysisInstance):
        """set the `analysis` instance

        Args:
            analysisInstance (pyCGM2.analysis.Analysis): an `analysis` instance
        """
        self.analysis = analysisInstance


    def export(self,outputName, path=None):
        """ Export as json

        Args:
            outputName (str): filename without extension
            path (str): Path
        """

        out=OrderedDict()
        out["Stp"]=OrderedDict()
        out["Kinematics"]=OrderedDict()
        out["Kinetics"]=OrderedDict()
        out["Emg"]=OrderedDict()
        out["Scores"]=OrderedDict()
        out["Scores"]["GPS"]=OrderedDict()
        out["Scores"]["GVS"]=OrderedDict()

        if self.analysis.stpStats != {} and self.analysis.stpStats is not None:
            processedKeys=list()
            for keys in self.analysis.stpStats.keys():
                if keys not in processedKeys:
                    processedKeys.append(keys)
                else:
                    raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                if keys[0] not in out["Stp"].keys():
                    out["Stp"][keys[0]]=dict()
                    out["Stp"][keys[0]][keys[1]]=dict()
                    out["Stp"][keys[0]][keys[1]]["values"]=self.analysis.stpStats[keys]["values"].tolist()
                else:
                    out["Stp"][keys[0]][keys[1]]=dict()
                    out["Stp"][keys[0]][keys[1]]["values"]=self.analysis.stpStats[keys]["values"].tolist()

        if self.analysis.kinematicStats.data != {}:
            processedKeys=list()
            for keys in self.analysis.kinematicStats.data.keys():
                if not np.all( self.analysis.kinematicStats.data[keys]["mean"]==0):

                    if keys not in processedKeys:
                        processedKeys.append(keys)
                    else:
                        raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                    if keys[0] not in out["Kinematics"].keys():
                        out["Kinematics"][keys[0]]=dict()
                        out["Kinematics"][keys[0]][keys[1]]=dict()
                        out["Kinematics"][keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}
                    else:
                        out["Kinematics"][keys[0]][keys[1]]=dict()
                        out["Kinematics"][keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}

                    li_X = list()
                    for cycle in self.analysis.kinematicStats.data[keys]["values"]:
                        li_X.append(cycle[:,0].tolist())

                    li_Y = list()
                    for cycle in self.analysis.kinematicStats.data[keys]["values"]:
                        li_Y.append(cycle[:,1].tolist())

                    li_Z = list()
                    for cycle in self.analysis.kinematicStats.data[keys]["values"]:
                        li_Z.append(cycle[:,2].tolist())


                    out["Kinematics"][keys[0]][keys[1]]["values"]["X"] = li_X
                    out["Kinematics"][keys[0]][keys[1]]["values"]["Y"] = li_Y
                    out["Kinematics"][keys[0]][keys[1]]["values"]["Z"] = li_Z

        if self.analysis.kineticStats.data != {}:
            processedKeys=list()
            for keys in self.analysis.kineticStats.data.keys():
                if not np.all( self.analysis.kineticStats.data[keys]["mean"]==0):

                    if keys not in processedKeys:
                        processedKeys.append(keys)
                    else:
                        raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                    if keys[0] not in out["Kinetics"].keys():
                        out["Kinetics"][keys[0]]=dict()
                        out["Kinetics"][keys[0]][keys[1]]=dict()
                        out["Kinetics"][keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}
                    else:
                        out["Kinetics"][keys[0]][keys[1]]=dict()
                        out["Kinetics"][keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}

                    li_X = list()
                    for cycle in self.analysis.kineticStats.data[keys]["values"]:
                        li_X.append(cycle[:,0].tolist())

                    li_Y = list()
                    for cycle in self.analysis.kineticStats.data[keys]["values"]:
                        li_Y.append(cycle[:,1].tolist())

                    li_Z = list()
                    for cycle in self.analysis.kineticStats.data[keys]["values"]:
                        li_Z.append(cycle[:,2].tolist())


                    out["Kinetics"][keys[0]][keys[1]]["values"]["X"] = li_X
                    out["Kinetics"][keys[0]][keys[1]]["values"]["Y"] = li_Y
                    out["Kinetics"][keys[0]][keys[1]]["values"]["Z"] = li_Z


        if self.analysis.emgStats.data != {}:
            processedKeys=list()
            for keys in self.analysis.emgStats.data.keys():
                if not np.all( self.analysis.emgStats.data[keys]["mean"]==0):
                    if keys not in processedKeys:
                        processedKeys.append(keys)
                    else:
                        raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                    key = keys[0][keys[0].rfind(".")+1:] if "Voltage." in keys[0] else keys[0]
                    if key not in out["Emg"].keys():
                        out["Emg"][key]=dict()
                        out["Emg"][key][keys[1]]=dict()
                        out["Emg"][key][keys[1]]["values"]=[]
                    else:
                        out["Emg"][key][keys[1]]=dict()
                        out["Emg"][key][keys[1]]["values"]=[]

                    li = list()
                    for cycle in self.analysis.emgStats.data[keys]["values"]:
                        li.append(cycle[:,0].tolist())

                    out["Emg"][key][keys[1]]["values"] = li

        if self.analysis.gvs != {}:
            processedKeys=list()
            for keys in self.analysis.gvs.keys():
                if not np.all( self.analysis.gvs[keys]["mean"]==0):

                    if keys not in processedKeys:
                        processedKeys.append(keys)
                    else:
                        raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                    if keys[0] not in out["Scores"]["GVS"].keys():
                        out["Scores"]["GVS"][keys[0]]=dict()
                        out["Scores"]["GVS"][keys[0]][keys[1]]=dict()
                        out["Scores"]["GVS"][keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}
                    else:
                        out["Scores"]["GVS"][keys[0]][keys[1]]=dict()
                        out["Scores"]["GVS"][keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}

                    li_X = list()
                    li_Y = list()
                    li_Z = list()
                    for cycleIndex in range(0,self.analysis.gvs[keys]["values"].shape[0]):
                        li_X.append(self.analysis.gvs[keys]["values"][cycleIndex,0])
                        li_Y.append(self.analysis.gvs[keys]["values"][cycleIndex,1])
                        li_Z.append(self.analysis.gvs[keys]["values"][cycleIndex,2])

                    out["Scores"]["GVS"][keys[0]][keys[1]]["values"]["X"] = li_X
                    out["Scores"]["GVS"][keys[0]][keys[1]]["values"]["Y"] = li_Y
                    out["Scores"]["GVS"][keys[0]][keys[1]]["values"]["Z"] = li_Z
        if self.analysis.gps != {}:

            out["Scores"]["GPS"]["mean"] = self.analysis.gps["Overall"]["mean"][0]
            out["Scores"]["GPS"]["std"] = self.analysis.gps["Overall"]["std"][0]
            out["Scores"]["GPS"]["median"] = self.analysis.gps["Overall"]["median"][0]
            out["Scores"]["GPS"]["values"] = self.analysis.gps["Overall"]["values"].tolist()


        files.saveJson(path,outputName,out)

        return out



# class AnalysisC3dExportFilter(object):
#     """
#          Filter exporting Analysis instance in json
#     """
#
#     def __init__(self):
#
#         self.analysis = None
#
#     def setAnalysisInstance(self,analysisInstance):
#         self.analysis = analysisInstance
#
#     # def _build(self,data):
#
#     def export(self,outputName, path=None):
#
#         root = ma.Node('root')
#         trial = ma.Trial("AnalysisC3d",root)
#
#         # metadata
#         #-------------
#
#         # subject infos
#         if self.analysis.subjectInfo is not None:
#             subjInfo = self.analysis.subjectInfo
#             for item in subjInfo.items():
#                 trial.setProperty("SUBJECT_INFO:"+ item[0],item[1])
#
#         # model infos
#         if self.analysis.modelInfo is not None:
#             modelInfo =  self.analysis.modelInfo
#             for item in modelInfo.items():
#                 trial.setProperty("MODEL_INFO:"+ item[0],item[1])
#
#         # model infos
#         if self.analysis.experimentalInfo is not None:
#             experimentalConditionInfo = self.analysis.experimentalInfo
#             for item in experimentalConditionInfo.items():
#                 trial.setProperty("EXPERIMENTAL_INFO:"+ item[0],item[1])
#
#
#         #trial.setProperty('MY_GROUP:MY_PARAMETER',10.0)
#
#         # kinematic cycles
#         #------------------
#
#         # metadata
#         # for key in self.analysis.kinematicStats.data.keys():
#         #     if key[1]=="Left":
#         #         n_left_cycle = len(self.analysis.kinematicStats.data[key[0],key[1]]["values"])
#         #         trial.setProperty('PROCESSING:LeftKinematicCycleNumber',n_left_cycle)
#         #         break
#         #
#         # for key in self.analysis.kinematicStats.data.keys():
#         #     if key[1]=="Right":
#         #         n_right_cycle = len(self.analysis.kinematicStats.data[key[0],key[1]]["values"])
#         #         trial.setProperty('PROCESSING:RightKinematicCycleNumber',n_right_cycle)
#         #         break
#
#         # cycles
#         for key in self.analysis.kinematicStats.data.keys():
#             label = key[0]
#             context = key[1]
#             if not np.all( self.analysis.kinematicStats.data[label,context]["mean"]==0):
#                 cycle = 0
#                 values = np.zeros((101,4))
#                 values2 = np.zeros((101,1))
#                 for val in self.analysis.kinematicStats.data[label,context]["values"]:
#                     angle = ma.TimeSequence(label+"."+context+"."+str(cycle),4,101,1.0,0.0,ma.TimeSequence.Type_Angle,"deg", trial.timeSequences())
#                     values[:,0:3] = val
#                     angle.setData(values)
#                     cycle+=1
#
#         # kinetic cycles
#         #------------------
#
#         # # metadata
#         # for key in self.analysis.kineticStats.data.keys():
#         #     if key[1]=="Left":
#         #         n_left_cycle = len(self.analysis.kineticStats.data[key[0],key[1]]["values"])
#         #         trial.setProperty('PROCESSING:LeftKineticCycleNumber',n_left_cycle)
#         #         break
#         #
#         # for key in self.analysis.kineticStats.data.keys():
#         #     if key[1]=="Right":
#         #         n_right_cycle = len(self.analysis.kineticStats.data[key[0],key[1]]["values"])
#         #         trial.setProperty('PROCESSING:RightKineticCycleNumber',n_right_cycle)
#         #         break
#
#         # cycles
#         for key in self.analysis.kineticStats.data.keys():
#             label = key[0]
#             context = key[1]
#             if not np.all( self.analysis.kineticStats.data[label,context]["mean"]==0):
#                 cycle = 0
#                 values = np.zeros((101,4))
#                 for val in self.analysis.kineticStats.data[label,context]["values"]:
#                     moment = ma.TimeSequence(str(label+"."+context+"."+str(cycle)),4,101,1.0,0.0,ma.TimeSequence.Type_Moment,"N.mm", trial.timeSequences())
#                     values[:,0:3] = val
#                     moment.setData(values)
#                     cycle+=1
#
#         # for key in self.analysis.emgStats.data.keys():
#         #     label = key[0]
#         #     context = key[1]
#         #     if not np.all( self.analysis.emgStats.data[label,context]["mean"]==0):
#         #         cycle = 0
#         #         for val in self.analysis.emgStats.data[label,context]["values"]:
#         #             analog = ma.TimeSequence(str(label+"."+context+"."+str(cycle)),1,101,1.0,0.0,ma.TimeSequence.Type_Analog,"V", 1.0,0.0,[-10.0,10.0], trial.timeSequences())
#         #             analog.setData(val)
#         #             cycle+=1
#
#
#         try:
#             if path == None:
#                 ma.io.write(root,outputName+".c3d" )
#             else:
#                 ma.io.write(root,path + outputName+".c3d")
#             LOGGER.logger.info("Analysis c3d  [%s.c3d] Exported" %( (outputName +".c3d")) )
#         except:
#             raise Exception ("[pyCGM2] : analysis c3d doesn t export" )
