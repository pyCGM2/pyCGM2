# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_flow.py::Test_flow
from jinja2 import Template

import pyCGM2
from pyCGM2.flow import flowFilters
from pyCGM2.flow.Nantes import eclipseFlowProcedure

from pyCGM2.Mek import mekTools


class Test_flow:
    def test_empty(self):

        template = pyCGM2.PYCGM2_SETTINGS_FOLDER+"templates\\cgm#i-settings.tpl"

        data = {
            "Patient": {},
            "Visit": {},
            "Mp": {},
            "Calibration": [],
            "Fitting": [],
            "Emg": [],
            "Conditions": []
        }
        jinja2_template_string = open(template, 'rb').read()
        template = Template(jinja2_template_string.decode("utf-8"))
        template.stream(data=data).dump("verif.settings")


    def test_emptyWithFilters(self):
        
        fef = flowFilters.FlowEdittingFilter(None,"CGM2.5")
        fef.run("CGMversion25.settings")


    def test_CGM21WithFilters(self):
        path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\flow\\Nantes\\MAIGNAN Olympe\\Session 1\\"

        fef = flowFilters.FlowEdittingFilter(path,"CGM2.1", procedure=eclipseFlowProcedure.EclipseFlowProcedure())
        fef.run("CGMversion21-verif.settings")












