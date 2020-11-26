# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
from  collections import OrderedDict

# pyCGM2
import pyCGM2
from pyCGM2.Utils import files
from pyCGM2.Tools import exportTools

def renameEmgInAnalysis(analysisInstance,emgChannels, emgMuscles, emgContexts):

    i=len(emgChannels)-1
    for channelIt in reversed(emgChannels):
        newlabel = emgContexts[i][0] + emgMuscles[i]
        for keyIt in analysisInstance.emgStats.data.keys():
            context = keyIt[1]
            if channelIt in keyIt[0]:
                newLabelFinal = keyIt[0].replace(channelIt,newlabel)
                analysisInstance.emgStats.data[newLabelFinal,context] = analysisInstance.emgStats.data.pop((keyIt[0],context))
                logging.debug("label [%s] replaced with [%s]"%(keyIt[0],newLabelFinal))
        i=i-1






class XlsExportDataFrameFilter(object):
    """
         Filter exporting Analysis instance in xls table
    """

    def __init__(self):

        self.dataframes =list()

    def setDataFrames(self, dataframes):

        if isinstance(dataframes,pd.core.frame.DataFrame):
            dataframes=[dataframes]

        for it in dataframes:
            self.dataframes.append(it)

    def export(self,outputName, path=None,excelFormat = "xls"):
        """

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
    """
         Filter exporting Analysis instance in xls table
    """

    def __init__(self):

        self.analysis = None

    def setAnalysisInstance(self,analysisInstance):
        self.analysis = analysisInstance

    def export(self,outputName, path=None,excelFormat = "xls",mode="Advanced"):
        if mode == "Advanced":
            self.__advancedExport(outputName, path=path, excelFormat = excelFormat)
        elif mode == "Basic":
            self.__basicExport(outputName, path=path, excelFormat = excelFormat)

    def __basicExport(self,outputName, path=None,excelFormat = "xls"):
        """
            export  member *analysis* as xls file in a basic mode.
            A basic xls puts Frame number in column. Each outputs is included as new sheet.

            :Parameters:
                - `outputName` (str) - name of the xls file ( without xls extension)
                - `path` (str) - folder in which xls files will be stored
                - `excelFormat` (str) - by default : xls. xlsx is also available

        """



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
        logging.info("basic dataFrame [%s- basic] Exported"%outputName)





    def __advancedExport(self,outputName, path=None, excelFormat = "xls",csvFileExport =False):
        """
            export  member *analysis* as xls file in a Advanced mode.
            A Advanced xls report outputs in a single sheet and put frames in row.

            .. note::

                an advanced xls contains the folowing sheets:
                    * `descriptive stp` : descriptive statistic of spatio-tenporal parameters
                    * `stp cycles` : all cycles used for computing descriptive statistics of spatio-temporal parameters
                    * `descriptive kinematics` : descriptive statistic of kinematic parameters
                    * `kinematics cycles` : all cycles used for computing descriptive statistics of kinematic parameters
                    * `descriptive kinetics` : descriptive statistic of kinetic parameters
                    * `kinetics cycles` : all cycles used for computing descriptive statistics of kinetic parameters

            :Parameters:
                - `outputName` (str) - name of the xls file ( without xls extension)
                - `path` (str) - folder in which xls files will be stored
                - `excelFormat` (str) - by default : xls. xlsx is also available
                - `csvFileExport` (bool) - enable export of csv files

        """


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
            df_descriptiveStp = exportTools.buid_df_descriptiveCycle1_1(self.analysis.stpStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveStp, key)
                    df_descriptiveStp[key] = value
            df_descriptiveStp.to_excel(xlsxWriter,'descriptive stp',index=False)


            # stage 2 : get cycle values
            # --------------------------------
            df_stp = exportTools.buid_df_cycles1_1(self.analysis.stpStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_stp, key)
                    df_stp[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_stp, key)
                    df_stp[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_stp, key)
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

            exportTools.buid_df_cycles1_1_onlyContext(self.analysis.gps["Context"], "Gps")


            df_descriptiveGpsByContext = exportTools.buid_df_descriptiveCycle1_1_onlyContext(self.analysis.gps["Context"], "Gps")
            df_descriptiveGpsOverall = exportTools.buid_df_descriptiveCycle1_1_overall(self.analysis.gps["Overall"],"Gps")
            df_descriptiveGps =  pd.concat([df_descriptiveGpsOverall,df_descriptiveGpsByContext])

            df_allGpsByContext = exportTools.buid_df_cycles1_1_onlyContext(self.analysis.gps["Context"], "Gps")

            # add infos
            for itdf in [df_descriptiveGps,df_descriptiveGpsByContext,df_allGpsByContext]:
                if modelInfo is not None:
                    for key,value in modelInfo.items():
                        exportTools.isColumnNameExist( itdf, key)
                        itdf[key] = value

                if subjInfo is not None:
                    for key,value in subjInfo.items():
                        exportTools.isColumnNameExist( itdf, key)
                        itdf[key] = value
                if condExpInfo is not None:
                    for key,value in condExpInfo.items():
                        exportTools.isColumnNameExist( itdf, key)
                        itdf[key] = value


            df_descriptiveGps.to_excel(xlsxWriter,'descriptive GPS ',index=False)
            df_allGpsByContext.to_excel(xlsxWriter,'GPS cycles ',index=False)


        if self.analysis.gvs is not None:
            df_descriptiveGvs = exportTools.buid_df_descriptiveCycle1_3(self.analysis.gvs,"Gvs")
            df_allGvs = exportTools.buid_df_cycles1_3(self.analysis.gvs, "Gvs")

            # add infos
            for itdf in [df_descriptiveGvs,df_allGvs]:
                if modelInfo is not None:
                    for key,value in modelInfo.items():
                        exportTools.isColumnNameExist( itdf, key)
                        itdf[key] = value

                if subjInfo is not None:
                    for key,value in subjInfo.items():
                        exportTools.isColumnNameExist( itdf, key)
                        itdf[key] = value
                if condExpInfo is not None:
                    for key,value in condExpInfo.items():
                        exportTools.isColumnNameExist( itdf, key)
                        itdf[key] = value


            df_descriptiveGvs.to_excel(xlsxWriter,'descriptive GVS ',index=False)
            df_allGvs.to_excel(xlsxWriter,'GVS cycles ',index=False)


        # Kinematics ouput
        #---------------------------


        if self.analysis.kinematicStats.data!={}:

            # stage 1 : get descriptive data
            # --------------------------------
            df_descriptiveKinematics = exportTools.buid_df_descriptiveCycle101_3(self.analysis.kinematicStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinematics, key)
                    df_descriptiveKinematics[key] = value

            df_descriptiveKinematics.to_excel(xlsxWriter,'descriptive kinematics ',index=False)

            # stage 2 : get cycle values
            # --------------------------------

            # cycles
            df_kinematics =  exportTools.buid_df_cycles101_3(self.analysis.kinematicStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_kinematics, key)
                    df_kinematics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_kinematics, key)
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
            df_descriptiveKinetics = exportTools.buid_df_descriptiveCycle101_3(self.analysis.kineticStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value

            df_descriptiveKinetics.to_excel(xlsxWriter,'descriptive kinetics ',index=False)

            # stage 2 : get cycle values
            # --------------------------------

            # cycles
            df_kinetics =  exportTools.buid_df_cycles101_3(self.analysis.kineticStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_kinetics, key)
                    df_kinetics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_kinetics, key)
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
            df_descriptiveEMG = exportTools.buid_df_descriptiveCycle101_1(self.analysis.emgStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveEMG, key)
                    df_descriptiveEMG[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveEMG, key)
                    df_descriptiveEMG[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveEMG, key)
                    df_descriptiveEMG[key] = value

            df_descriptiveEMG.to_excel(xlsxWriter,'descriptive EMG ',index=False)

            # stage 2 : get cycle values
            # --------------------------------

            # cycles
            df_emg =  exportTools.buid_df_cycles101_1(self.analysis.emgStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_emg, key)
                    df_emg[key] = value

            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_emg, key)
                    df_emg[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_emg, key)
                    df_emg[key] = value

            df_emg.to_excel(xlsxWriter,'EMG cycles',index=False)
            if csvFileExport:
                if path == None:
                    df_emg.to_csv((outputName + " - EMG - DataFrame.csv"),sep=";")
                else:
                    df_emg.to_csv((path+outputName + " - EMG - DataFrame.csv"),sep=";")


        logging.info("advanced dataFrame [%s- Advanced] Exported"%outputName)

        xlsxWriter.save()


class AnalysisExportFilter(object):
    """
         Filter exporting Analysis instance in json
    """

    def __init__(self):

        self.analysis = None

    def setAnalysisInstance(self,analysisInstance):
        self.analysis = analysisInstance

    # def _build(self,data):

    def export(self,outputName, path=None):


        out=OrderedDict()

        if self.analysis.stpStats != {}:
            processedKeys=list()
            for keys in self.analysis.stpStats.keys():
                if keys not in processedKeys:
                    processedKeys.append(keys)
                else:
                    raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                if keys[0] not in out.keys():
                    out[keys[0]]=dict()
                    out[keys[0]][keys[1]]=dict()
                    out[keys[0]][keys[1]]["values"]=self.analysis.stpStats[keys]["values"].tolist()
                else:
                    out[keys[0]][keys[1]]=dict()
                    out[keys[0]][keys[1]]["values"]=self.analysis.stpStats[keys]["values"].tolist()

        if self.analysis.kinematicStats.data != {}:
            processedKeys=list()
            for keys in self.analysis.kinematicStats.data.keys():
                if not np.all( self.analysis.kinematicStats.data[keys]["mean"]==0):

                    if keys not in processedKeys:
                        processedKeys.append(keys)
                    else:
                        raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                    if keys[0] not in out.keys():
                        out[keys[0]]=dict()
                        out[keys[0]][keys[1]]=dict()
                        out[keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}
                    else:
                        out[keys[0]][keys[1]]=dict()
                        out[keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}

                    li_X = list()
                    for cycle in self.analysis.kinematicStats.data[keys]["values"]:
                        li_X.append(cycle[:,0].tolist())

                    li_Y = list()
                    for cycle in self.analysis.kinematicStats.data[keys]["values"]:
                        li_Y.append(cycle[:,1].tolist())

                    li_Z = list()
                    for cycle in self.analysis.kinematicStats.data[keys]["values"]:
                        li_Z.append(cycle[:,2].tolist())


                    out[keys[0]][keys[1]]["values"]["X"] = li_X
                    out[keys[0]][keys[1]]["values"]["Y"] = li_Y
                    out[keys[0]][keys[1]]["values"]["Z"] = li_Z

        if self.analysis.kineticStats.data != {}:
            processedKeys=list()
            for keys in self.analysis.kineticStats.data.keys():
                if not np.all( self.analysis.kineticStats.data[keys]["mean"]==0):

                    if keys not in processedKeys:
                        processedKeys.append(keys)
                    else:
                        raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                    if keys[0] not in out.keys():
                        out[keys[0]]=dict()
                        out[keys[0]][keys[1]]=dict()
                        out[keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}
                    else:
                        out[keys[0]][keys[1]]=dict()
                        out[keys[0]][keys[1]]["values"]= {"X":[],"Y":[],"Z":[]}

                    li_X = list()
                    for cycle in self.analysis.kineticStats.data[keys]["values"]:
                        li_X.append(cycle[:,0].tolist())

                    li_Y = list()
                    for cycle in self.analysis.kineticStats.data[keys]["values"]:
                        li_Y.append(cycle[:,1].tolist())

                    li_Z = list()
                    for cycle in self.analysis.kineticStats.data[keys]["values"]:
                        li_Z.append(cycle[:,2].tolist())


                    out[keys[0]][keys[1]]["values"]["X"] = li_X
                    out[keys[0]][keys[1]]["values"]["Y"] = li_Y
                    out[keys[0]][keys[1]]["values"]["Z"] = li_Z


        if self.analysis.emgStats.data != {}:
            processedKeys=list()
            for keys in self.analysis.emgStats.data.keys():
                if not np.all( self.analysis.emgStats.data[keys]["mean"]==0):
                    if keys not in processedKeys:
                        processedKeys.append(keys)
                    else:
                        raise Exception ( "[pyCGM2] - duplicated keys[ %s - %s] detected" %(keys[0],keys[1]))

                    if keys[0] not in out.keys():
                        out[keys[0]]=dict()
                        out[keys[0]][keys[1]]=dict()
                        out[keys[0]][keys[1]]["values"]=[]
                    else:
                        out[keys[0]][keys[1]]=dict()
                        out[keys[0]][keys[1]]["values"]=[]

                    li = list()
                    for cycle in self.analysis.emgStats.data[keys]["values"]:
                        li.append(cycle[:,0].tolist())

                    out[keys[0]][keys[1]]["values"] = li

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
#             logging.info("Analysis c3d  [%s.c3d] Exported" %( (outputName +".c3d")) )
#         except:
#             raise Exception ("[pyCGM2] : analysis c3d doesn t export" )
