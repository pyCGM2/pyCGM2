# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_flow.py::Test_flowApp::test_mekImport
from jinja2 import Template

import pyCGM2
from pyCGM2.flow import flowFilters
from pyCGM2.flow.procedures import eclipseFlowProcedure
from pyCGM2.Utils import files
from pyCGM2.Mek.mek import mekTools

from pyCGM2.flow import settingsHandler 
from pyCGM2.Apps.flow import flowInit
from pyCGM2.Apps.flow import flowEdit
from pyCGM2.Apps.flow import flowMekImporter
from pyCGM2.Apps.flow import flowPrepare
from pyCGM2.Apps.flow import flowMekPopulate



from argparse import Namespace
    

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


    def test_CGM24WithFilters2Conditions(self):
        path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\flow\\Nantes\\ESNAULT Oceane\\Session 3\\"

        previousEmgSttings = files.openFile(path, "emg.settings")


        fef = flowFilters.FlowEdittingFilter(path,"CGM2.4", 
                                             procedure=eclipseFlowProcedure.EclipseFlowProcedure(),
                                             emgSettings = None
                                             )

        fef.run("CGMversion24-verif.settings")



        settings = files.openFile(path, "CGMversion24-verif.settings")



        # emgSettings = files.openFile(path,"emg.settings")
        settings["Protocol"]["Conditions"][0]["EmgSettings"]
        # files.saveYaml(path,"test.yaml",settings)
        import ipdb; ipdb.set_trace()




class Test_flowApp:
    def test_init(self):
        path = pyCGM2.TEST_DATA_PATH + "Nantes\\OSMANOV Akhmed\\Session 3\\"
        args = Namespace(  subparser="FLOW" ,  FLOW="Init",         data_path=path    )
        flowInit.main(args=args) 

    def test_edit(self):
        path = pyCGM2.TEST_DATA_PATH + "Nantes\\OSMANOV Akhmed\\Session 3\\"
        args = Namespace(  subparser="FLOW" ,  FLOW="Edit",         
                         cgmVersion="CGM2.3", suffix="newtest", display=True,data_path=path    )
        flowEdit.main(args=args) 
    
    def test_mekImport(self):
        path = pyCGM2.TEST_DATA_PATH + "NantesSamples\\AQM Adultes\\BOUCHE Alain\\Session 1\\"
        args = Namespace(  subparser="FLOW" ,  FLOW="Import",         
                         userSettings="CGM23_v2", data_path=path,
                        conditions=None )
        
        flowMekImporter.main(args=args) 


    def test_prepare(self):
        path = pyCGM2.TEST_DATA_PATH + "NantesSamples\\AQM Adultes\\BOUCHE Alain\\Session 1\\"
        args = Namespace(  subparser="FLOW" ,  FLOW="Prepare",         
                         userSettings="CGM23_v2", data_path=path,
                         conditions=None )
        flowPrepare.main(args=args) 




    def test_mekPopulate(self):
        path = pyCGM2.TEST_DATA_PATH + "NantesSamples\\AQM Adultes\\BOUCHE Alain\\Session 1\\"
        args = Namespace(  subparser="FLOW" ,  FLOW="Populate",         
                         userSettings="CGM23_v2", data_path=path,
                         analysisID =None,
                         update=True,
                        conditions=None )
        flowMekPopulate.main(args=args) 













