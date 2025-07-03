# -*- coding: utf-8 -*-
import os
import pyCGM2
from jinja2 import Template

class FlowEdittingFilter(object):

    def __init__(self,data_path, modelVersion,procedure=None):
        self.m_template = pyCGM2.PYCGM2_SETTINGS_FOLDER+"templates\\cgm#i-settings.tpl"

        self.m_procedure = procedure

        self.m_modelVersion = modelVersion
        self.m_data_path = data_path

    def setTemplate(self, template_pathFilename):
        self.m_template = template_pathFilename

    def restoreInitialTemplate(self):
        self.m_template = pyCGM2.PYCGM2_SETTINGS_FOLDER+"templates\\cgm#i-settings.tpl"


    def run(self, filename):

        if self.m_procedure is not None:
            data = self.m_procedure.run(self.m_data_path,self.m_modelVersion)
        else: 
            data = {
                "ModelVersion": self.m_modelVersion,
                "Patient": {},
                "Visit": {},
                "Mp": {},
                "Calibration": [],
                "Fitting": [],
                "Emg": [],
                "Conditions": []
            }
        if self.m_data_path is None : self.m_data_path=""

        jinja2_template_string = open(self.m_template, 'rb').read()
        template = Template(jinja2_template_string.decode("utf-8"))
        template.stream(data=data).dump(self.m_data_path+filename)
        os.startfile(self.m_data_path+filename)
