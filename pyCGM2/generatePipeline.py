# -*- coding: utf-8 -*-
import string
import os

CALIBRATION_CONTENT ="""<?xml version="1.1" encoding="UTF-8" standalone="no" ?>
    <Pipeline>

      <Entry DisplayName="Run pyCGM2-iMODEL- Calibration " Enabled="1" OperationId="104" OperationName="Python">
        <ParamList name="">
          <Param name="Script" value="PATH_APPS/iMODEL/nexusOperation-pyCGM2-iMODEL-Calibration.py"/>
          <Param name="ScriptArgs"/>
          <Param name="UseNexusPython" value="false"/>
          <Param name="LaunchPython" value="false"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="Save Trial - C3D + VSK" Enabled="1" OperationId="105" OperationName="SaveOperation">
        <ParamList name="">
          <Param macro="SELECTED_START_FRAME" name="StartFrame"/>
          <Param macro="SELECTED_END_FRAME" name="EndFrame"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="add pyCGM2-iMODEL- metadata" Enabled="1" OperationId="106" OperationName="Python">
        <ParamList name="">
          <Param name="Script" value="PATH_APPS/iMODEL/nexusOperation_pyCGM2-iMODEL-metadata.py"/>
          <Param name="ScriptArgs" value="--calibration"/>
          <Param name="UseNexusPython" value="false"/>
          <Param name="LaunchPython" value="false"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="Run pyCGM2-CGM1-i- static Processing " Enabled="1" OperationId="107" OperationName="Python">
        <ParamList name="">
          <Param name="Script" value="PATH_APPS/CGM1-i_dataProcessing/nexusOperation-pyCGM2-CGM1_i-staticProcessing.py"/>
          <Param name="ScriptArgs"/>
          <Param name="UseNexusPython" value="false"/>
          <Param name="LaunchPython" value="false"/>
        </ParamList>
      </Entry>

    </Pipeline>"""



FITTING_CONTENT ="""<?xml version="1.1" encoding="UTF-8" standalone="no" ?>
    <Pipeline>

      <Entry DisplayName="Run pyCGM2-iMODEL- Fitting " Enabled="1" OperationId="104" OperationName="Python">
        <ParamList name="">
          <Param name="Script" value="PATH_APPS/iMODEL/nexusOperation-pyCGM2-iMODEL-Fitting.py"/>
          <Param name="ScriptArgs"/>
          <Param name="UseNexusPython" value="false"/>
          <Param name="LaunchPython" value="false"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="Save Trial - C3D + VSK" Enabled="1" OperationId="105" OperationName="SaveOperation">
        <ParamList name="">
          <Param macro="SELECTED_START_FRAME" name="StartFrame"/>
          <Param macro="SELECTED_END_FRAME" name="EndFrame"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="add pyCGM2-iMODEL- metadata" Enabled="1" OperationId="106" OperationName="Python">
        <ParamList name="">
          <Param name="Script" value="PATH_APPS/iMODEL/nexusOperation_pyCGM2-iMODEL-metadata.py"/>
          <Param name="ScriptArgs"/>
          <Param name="UseNexusPython" value="false"/>
          <Param name="LaunchPython" value="false"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="Run pyCGM2-CGM1-i- Gait Processing " Enabled="1" OperationId="107" OperationName="Python">
        <ParamList name="">
          <Param name="Script" value="PATH_APPS/CGM1-i_dataProcessing/nexusOperation-pyCGM2-CGM1_i-gaitProcessing.py"/>
          <Param name="ScriptArgs"/>
          <Param name="UseNexusPython" value="false"/>
          <Param name="LaunchPython" value="false"/>
        </ParamList>
      </Entry>

    </Pipeline>"""


# ------------------- CGM1 ------------------------------------------------------
def pipeline_pyCGM2_CGM1_Calibration(myAppFolder_path):

    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM1")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( str(myAppFolder_path+"CGM1\\") + "pyCGM2-CGM1-Calibration.Pipeline"):
        with open(str(myAppFolder_path+"CGM1\\") + "pyCGM2-CGM1-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)
            
def pipeline_pyCGM2_CGM1_Fitting(myAppFolder_path):

    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM1")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( str(myAppFolder_path+"CGM1\\") + "pyCGM2-CGM1-Fitting.Pipeline"):
        with open(str(myAppFolder_path+"CGM1\\") + "pyCGM2-CGM1-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)     

#-----------------------CGM 1.1------------------------------------------------            
def pipeline_pyCGM2_CGM1_1_Calibration(myAppFolder_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM1_1")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( str(myAppFolder_path+"CGM1_1\\") + "pyCGM2-CGM1_1-Calibration.Pipeline"):
        with open(str(myAppFolder_path+"CGM1_1\\") + "pyCGM2-CGM1_1-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)
            
def pipeline_pyCGM2_CGM1_1_Fitting(myAppFolder_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM1_1")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( str(myAppFolder_path+"CGM1_1\\") + "pyCGM2-CGM1_1-Fitting.Pipeline"):
        with open(str(myAppFolder_path+"CGM1_1\\") + "pyCGM2-CGM1_1-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)   