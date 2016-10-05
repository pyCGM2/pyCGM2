# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 16:17:08 2016

@author: fabien Leboeuf
"""
import numpy as np
import pandas as pd   
import logging
import pdb

# openMA
import ma.io
import ma.body


# ----- PANDAS ---------
# TODO : programmation a optimiser

FRAMES_HEADER= ["Frame"+str(i) for i in range(0,101)]

def isColumnNameExist( dataframe, name):
    if name in dataframe.columns.values:
        logging.warning("column name : %s already in the dataFrame",name)


def buid_df_descriptiveCycle1_1(analysis_outputStats):

    df_collection_L={"mean" : [], "std" : [], "median" : []}
    df_collection_R={"mean" : [], "std" : [], "median" : []}
    
    for key in analysis_outputStats.keys():
        
        label = key[0]
        context = key[1]
        
        if context == "Left":
            for typIt in ["mean","std","median"]:
                df_L = pd.DataFrame({ label : [ analysis_outputStats[label,context][str(typIt)] ],
                                         'Context': str(context),
                                         'Stats type' : str(typIt) }) 
                df_collection_L[typIt].append(df_L)
        
        
        if context == "Right":
            for typIt in ["mean","std","median"]:
                df_R = pd.DataFrame({ label : [ analysis_outputStats[label,context][str(typIt)] ],
                                   'Context': str(context),
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


def buid_df_cycles1_1(analysis_outputStats):

    df_collection_L=[]
    df_collection_R=[]
    
    for key in analysis_outputStats.keys():
        label = key[0]
        context = key[1]
        n =len(analysis_outputStats[label,context]['values'])            
        
        if context == "Left":
            df_L = pd.DataFrame({ label : analysis_outputStats[label,context]['values'],
                               "Cycle": range(0,n),
                               'Context': str(context) 
                               }) 
            
            df_collection_L.append(df_L)
        
 
        if context == "Right":
            df_R = pd.DataFrame({ label : analysis_outputStats[label,context]['values'],
                               "Cycle": range(0,n),
                               'Context': str(context) 
                               }) 
            
            df_collection_R.append(df_R)     

    left_flag = len(df_collection_L)
    right_flag = len(df_collection_R)
 
    if left_flag:  
        df_L=df_collection_L[0]       
        for i in range(1,len(df_collection_L)):
            df_L=pd.merge(df_L,df_collection_L[i])        
    
    if right_flag: 
        df_pst_R=df_collection_R[0]       
        for i in range(1,len(df_collection_R)):
            df_pst_R=pd.merge(df_pst_R,df_collection_R[i])
    
    if left_flag and right_flag:
        DF = pd.concat([df_L,df_R],ignore_index=True)
    elif left_flag and not right_flag:
        DF = df_L
    elif not left_flag and  right_flag:
       DF = df_R
           
    return DF






def buid_df_descriptiveCycle101_3(analysis_outputStats):


    # "data" section 
    df_collection_L=[]
    df_collection_R=[]
    
    for key in analysis_outputStats.data.keys():
        label = key[0]
        context = key[1]        
        if context == "Left":
            valuesMean=analysis_outputStats.data[label,context]["mean"]
            df=pd.DataFrame(valuesMean.T,  columns = FRAMES_HEADER)
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['Context']= context
            df['Stats type'] = "mean"
            df_collection_L.append(df)

            valuesStd=analysis_outputStats.data[label,context]["std"]
            df=pd.DataFrame(valuesStd.T,  columns= FRAMES_HEADER)
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['Context']= context
            df['Stats type'] = "std"
            df_collection_L.append(df)

            valuesMedian=analysis_outputStats.data[label,context]["median"]
            df=pd.DataFrame(valuesMedian.T,  columns= FRAMES_HEADER)
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['Context']= context
            df['Stats type'] = "median"
            df_collection_L.append(df)


        if context == "Right":
            valuesMean=analysis_outputStats.data[label,context]["mean"]
            df=pd.DataFrame(valuesMean.T,  columns= FRAMES_HEADER)
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['Context']= context
            df['Stats type'] = "mean"
            df_collection_R.append(df)

            valuesStd=analysis_outputStats.data[label,context]["std"]
            df=pd.DataFrame(valuesStd.T,  columns= FRAMES_HEADER)
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['Context']= context
            df['Stats type'] = "std"
            df_collection_R.append(df)

            valuesMedian=analysis_outputStats.data[label,context],["median"]
            df=pd.DataFrame(valuesMedian.T,  columns= FRAMES_HEADER)
            df['Axe']=['X','Y','Z']
            df['Label']=label
            df['Context']= context
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




def buid_df_cycles101_3(analysis_outputStats):

    # "data" section 
    df_collection_L=[]
    df_collection_R=[]
    
    for key in analysis_outputStats.data.keys():
        label = key[0]
        context = key[1]        
        i_l=0
        i_r=0
        if context =="Left" : # FIXME TODO : if different form L and R
            for itValues in analysis_outputStats.data[label,context]["values"]:
                df=pd.DataFrame(itValues.T,  columns= FRAMES_HEADER)
                df['Axis']=['X','Y','Z']
                df['Label']=label
                df['Cycle']= i_l
                df['Context']= context
                df_collection_L.append(df)
                i_l+=1

        if context=="Right" :
            for itValues in analysis_outputStats.data[label,context]["values"]:
                df=pd.DataFrame(itValues.T,  columns= FRAMES_HEADER)
                df['Axis']=['X','Y','Z']
                df['Label']=label
                df['Cycle']= i_r
                df['Context']= context
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
                                   'Context': str(context) 
                                   }) 
                
                df_collection_pst_L.append(df_pst_L)
     
            if context == "Right":
                df_pst_R = pd.DataFrame({ label : analysis_outputStats.pst[label,context]['values'],
                                   "Cycle": range(0,n),
                                   'Context': str(context) 
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
        DF=pd.merge(df_cycles,df_pst, on=['Cycle','Context'],how='outer') 

    else:
        DF = df_cycles


    return DF






