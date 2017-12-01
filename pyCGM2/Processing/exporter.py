# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging

# pyCGM2
import pyCGM2.Processing.cycle as CGM2cycle
from pyCGM2.Tools import exportTools

# openMA
import ma.io
import ma.body

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
                    xlsxWriter = pd.ExcelWriter(str(outputName + "- dataframe.xls"),engine='xlwt')
                elif excelFormat == "xlsx":
                    xlsxWriter = pd.ExcelWriter(str(outputName + "- dataframe.xlsx"))
            else:
                if excelFormat == "xls":
                    xlsxWriter = pd.ExcelWriter(str(path+outputName + "- dataframe.xls"),engine='xlwt')
                elif excelFormat == "xlsx":
                    xlsxWriter = pd.ExcelWriter(str(path+outputName + "- dataFrame.xlsx"))

            dataframe.to_excel(xlsxWriter,"dataframe_"+str(i))
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
                xlsxWriter = pd.ExcelWriter(str(outputName + "- basic.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter(str(outputName + "- basic.xlsx"))
        else:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter(str(path+outputName + "- basic.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter(str(path+outputName + "- basic.xlsx"))

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


        df_metadata.to_excel(xlsxWriter,"Infos")




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
                df_pst_L.to_excel(xlsxWriter,"Left - stp-kinematics")

            if dfs_r !=[]:
                df_pst_R=pd.concat(dfs_r)
                df_pst_R.to_excel(xlsxWriter,"Right - stp-kinematics")


            # kinematic cycles
            for key in self.analysis.kinematicStats.data.keys():
                label = key[0]
                context = key[1]
                n = len(self.analysis.kinematicStats.data[label,context]["values"])
                X = np.zeros((101,n))
                Y = np.zeros((101,n))
                Z = np.zeros((101,n))
                for i in range(0,n):
                    X[:,i] = self.analysis.kinematicStats.data[label,context]["values"][i][:,0]
                    Y[:,i] = self.analysis.kinematicStats.data[label,context]["values"][i][:,1]
                    Z[:,i] = self.analysis.kinematicStats.data[label,context]["values"][i][:,2]

                cycle_header= ["Cycle "+str(i) for i in range(0,n)]
                frame_header= ["Frame "+str(i) for i in range(0,101)]


                df_x=pd.DataFrame(X,  columns= cycle_header,index = frame_header )
                df_x['Axis']='X'
                df_y=pd.DataFrame(Y,  columns= cycle_header,index = frame_header )
                df_y['Axis']='Y'
                df_z=pd.DataFrame(Z,  columns= cycle_header,index = frame_header )
                df_z['Axis']='Z'

                df_label = pd.concat([df_x,df_y,df_z])
                df_label.to_excel(xlsxWriter,str(label+"."+context))

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
                df_pst_L.to_excel(xlsxWriter,"Left - pst-kinetics")

            if dfs_r !=[]:
                df_pst_R=pd.concat(dfs_r)
                df_pst_R.to_excel(xlsxWriter,"Right - pst-kinetics")


            # kinetic cycles
            for key in self.analysis.kineticStats.data.keys():
                label=key[0]
                context=key[1]

                n = len(self.analysis.kineticStats.data[label,context]["values"])
                X = np.zeros((101,n))
                Y = np.zeros((101,n))
                Z = np.zeros((101,n))
                for i in range(0,n):
                    X[:,i] = self.analysis.kineticStats.data[label,context]["values"][i][:,0]
                    Y[:,i] = self.analysis.kineticStats.data[label,context]["values"][i][:,1]
                    Z[:,i] = self.analysis.kineticStats.data[label,context]["values"][i][:,2]

                cycle_header= ["Cycle "+str(i) for i in range(0,n)]
                frame_header= ["Frame "+str(i) for i in range(0,101)]

                df_x=pd.DataFrame(X,  columns= cycle_header,index = frame_header )
                df_x['Axis']='X'
                df_y=pd.DataFrame(Y,  columns= cycle_header,index = frame_header )
                df_y['Axis']='Y'
                df_z=pd.DataFrame(Z,  columns= cycle_header,index = frame_header )
                df_z['Axis']='Z'

                df_label = pd.concat([df_x,df_y,df_z])
                df_label.to_excel(xlsxWriter,str(label+"."+context))

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
                xlsxWriter = pd.ExcelWriter(str(outputName + "- Advanced.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter(str(outputName + "- Advanced.xlsx"))
        else:
            if excelFormat == "xls":
                xlsxWriter = pd.ExcelWriter(str(path+outputName + "- Advanced.xls"),engine='xlwt')
            elif excelFormat == "xlsx":
                xlsxWriter = pd.ExcelWriter(str(path+outputName + "- Advanced.xlsx"))

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
            df_descriptiveStp.to_excel(xlsxWriter,'descriptive stp')


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

            df_stp.to_excel(xlsxWriter,'stp cycles')

            if csvFileExport:
                if path == None:
                    df_stp.to_csv(str(outputName + " - stp - DataFrame.csv"),sep=";")
                else:
                    df_stp.to_csv(str(path+outputName + " - stp - DataFrame.csv"),sep=";")

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


            df_descriptiveGps.to_excel(xlsxWriter,'descriptive GPS ')
            df_allGpsByContext.to_excel(xlsxWriter,'GPS cycles ')


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


            df_descriptiveGvs.to_excel(xlsxWriter,'descriptive GVS ')
            df_allGvs.to_excel(xlsxWriter,'GVS cycles ')


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

            df_descriptiveKinematics.to_excel(xlsxWriter,'descriptive kinematics ')

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

            df_kinematics.to_excel(xlsxWriter,'Kinematic cycles')
            if csvFileExport:
                if path == None:
                    df_kinematics.to_csv(str(outputName + " - kinematics - DataFrame.csv"),sep=";")
                else:
                    df_kinematics.to_csv(str(path+outputName + " - kinematics - DataFrame.csv"),sep=";")



        # Kinetic ouputs
        #---------------------------
        if self.analysis.kineticStats.data!={}:

            # stage 1 : get descriptive data
            # --------------------------------
            df_descriptiveKinetics = exportTools.buid_df_descriptiveCycle101_3(self.analysis.kineticStats)

            # add infos
            if modelInfo is not None:
                for key,value in modelInfo.items():
                    exportTools.isColumnNameExist( df_stp, key)
                    df_descriptiveKinetics[key] = value
            if subjInfo is not None:
                for key,value in subjInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value
            if condExpInfo is not None:
                for key,value in condExpInfo.items():
                    exportTools.isColumnNameExist( df_descriptiveKinetics, key)
                    df_descriptiveKinetics[key] = value

            df_descriptiveKinetics.to_excel(xlsxWriter,'descriptive kinetics ')

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

            df_kinetics.to_excel(xlsxWriter,'Kinetic cycles')
            if csvFileExport:
                if path == None:
                    df_stp.to_csv(str(outputName + " - kinetics - DataFrame.csv"),sep=";")
                else:
                    df_stp.to_csv(str(path+outputName + " - kinetics - DataFrame.csv"),sep=";")

        logging.info("advanced dataFrame [%s- Advanced] Exported"%outputName)

        xlsxWriter.save()
