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


SARA_CONTENT="""<?xml version="1.1" encoding="UTF-8" standalone="no" ?>
    <Pipeline>
    
      <Entry DisplayName="Save Trial - C3D + VSK" Enabled="1" OperationId="49" OperationName="SaveOperation">
        <ParamList name="">
          <Param macro="SELECTED_START_FRAME" name="StartFrame"/>
          <Param macro="SELECTED_END_FRAME" name="EndFrame"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="Run Python Operation" Enabled="1" OperationId="50" OperationName="Python">
        <ParamList name="">
          <Param name="Script" value="PATH_APPS/CGM2_3p_kneeCalibration/nexusOperation-pyCGM2-CGM2_3p_SARA.py"/>
          <Param name="ScriptArgs"/>
          <Param name="UseNexusPython" value="false"/>
          <Param name="LaunchPython" value="false"/>
        </ParamList>
      </Entry>
    
      <Entry DisplayName="Save Trial - C3D + VSK" Enabled="1" OperationId="51" OperationName="SaveOperation">
        <ParamList name="">
          <Param macro="SELECTED_START_FRAME" name="StartFrame"/>
          <Param macro="SELECTED_END_FRAME" name="EndFrame"/>
        </ParamList>
      </Entry>
    
    </Pipeline>"""

# ------------------- CGM1 ------------------------------------------------------
def pipeline_pyCGM2_CGM1_Calibration(myAppFolder_path,userAppData_path):

    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM1")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM1-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM1-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)

            
def pipeline_pyCGM2_CGM1_Fitting(myAppFolder_path,userAppData_path):

    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM1")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM1-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM1-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)     
            

#-----------------------CGM 1.1------------------------------------------------            
def pipeline_pyCGM2_CGM1_1_Calibration(myAppFolder_path,userAppData_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM1_1")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM1_1-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM1_1-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)

    
            
def pipeline_pyCGM2_CGM1_1_Fitting(myAppFolder_path,userAppData_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM1_1")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM1_1-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM1_1-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new) 
            
            
#-----------------------CGM 2.1------------------------------------------------            
def pipeline_pyCGM2_CGM2_1_Calibration(myAppFolder_path,userAppData_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM2_1")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_1-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_1-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)

    
            
def pipeline_pyCGM2_CGM2_1_Fitting(myAppFolder_path,userAppData_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM2_1")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_1-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_1-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)      

#-----------------------CGM 2.2------------------------------------------------            
def pipeline_pyCGM2_CGM2_2_Calibration(myAppFolder_path,userAppData_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM2_2")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_2-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_2-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)

    
            
def pipeline_pyCGM2_CGM2_2_Fitting(myAppFolder_path,userAppData_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM2_2")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_2-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_2-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)
            
#-----------------------CGM 2.2 EXPERT------------------------------------------------            
def pipeline_pyCGM2_CGM2_2_Expert_Calibration(myAppFolder_path,userAppData_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM2_2-Expert")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_2-Expert-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_2-Expert-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)

    
            
def pipeline_pyCGM2_CGM2_2_Expert_Fitting(myAppFolder_path,userAppData_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM2_2-Expert")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_2-Expert-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_2-Expert-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)  

#-----------------------CGM 2.3------------------------------------------------            
def pipeline_pyCGM2_CGM2_3_Calibration(myAppFolder_path,userAppData_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM2_3")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_3-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_3-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)       
            
def pipeline_pyCGM2_CGM2_3_Fitting(myAppFolder_path,userAppData_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM2_3")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_3-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_3-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)
            
#-----------------------CGM 2.3 EXPERT------------------------------------------            
def pipeline_pyCGM2_CGM2_3_Expert_Calibration(myAppFolder_path,userAppData_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM2_3-Expert")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_3-Expert-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_3-Expert-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)       
            
def pipeline_pyCGM2_CGM2_3_Expert_Fitting(myAppFolder_path,userAppData_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM2_3-Expert")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_3-Expert-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_3-Expert-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)
            
#-----------------------CGM 2.3i -SARA method------------------------------------------            
def pipeline_pyCGM2_CGM2_3_SARA_kneeCalibration(myAppFolder_path,userAppData_path):
   
    
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(SARA_CONTENT, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_3p_SARA.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_3p_SARA.Pipeline", "w") as text_file:
            text_file.write(content)       
            
#-----------------------CGM 2.4------------------------------------------------            
def pipeline_pyCGM2_CGM2_4_Calibration(myAppFolder_path,userAppData_path):
   
    content = string.replace(CALIBRATION_CONTENT, 'iMODEL', "CGM2_4")
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])

    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_4-Calibration.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_4-Calibration.Pipeline", "w") as text_file:
            text_file.write(content_new)       
            
def pipeline_pyCGM2_CGM2_4_Fitting(myAppFolder_path,userAppData_path):

       
    myAppFolder_path_slash = string.replace(myAppFolder_path, '\\', '/')
    
    content = string.replace(FITTING_CONTENT, 'iMODEL', "CGM2_4")
    content_new = string.replace(content, 'PATH_APPS', myAppFolder_path_slash[:-1])
    
    if not os.path.isfile( userAppData_path + "pyCGM2-CGM2_4-Fitting.Pipeline"):
        with open(userAppData_path + "pyCGM2-CGM2_4-Fitting.Pipeline", "w") as text_file:
            text_file.write(content_new)