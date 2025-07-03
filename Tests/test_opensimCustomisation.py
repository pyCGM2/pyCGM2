# pytest -s --disable-pytest-warnings  test_opensimCustomisation.py::Test_opensimcustom::test_cgm23


import ipdb
import os
import matplotlib.pyplot as plt


import pyCGM2; LOGGER = pyCGM2.LOGGER

import pyCGM2


from pyCGM2.Utils import files
from pyCGM2.Tools import  btkTools

from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model import  modelFilters,modelDecorator

from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Nexus import vskTools

from pyCGM2.Model.Opensim.interface import opensimInterface
from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures.scaling import opensimScalingInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.analysisReport import opensimAnalysesInterfaceProcedure

class Test_opensimcustom:
    
    def test_cgm23(self):


        data_path = "C:\\Users\\fleboeuf\\Documents\\ANALYSES\\pyCGM2-Analyses\\musclePersonalisation\\data\\bah\\"
        #--- fichiers d 'entrée-----
        staticFilename = "BAH Mariama Cal 01.c3d" 
        gaitFilename ="20240306-MB-PRE-S-NNNN-dyn04.c3d"


        #----Appel des settings----
        modelVersion = "CGM2.3"
        settings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")

        translators = settings["Translators"]
        weights = settings["Fitting"]["Weight"]
        hjcMethod = settings["Calibration"]["HJC"]

        markerDiameter=14


        #--- recuperation des données anthropometriques----

        # possibilité 1 - appel du fichier vsk 
        vsk = vskTools.Vsk(data_path + "BAH Mariama.vsk")
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)

        # --- Calibration du modele CGM2.3 ---

        acqStatic = btkTools.smartReader(data_path +  staticFilename) # construction btkAcquisition instance

        trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
        actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)


        # configuration du modele
        dcm = cgm.CGM.detectCalibrationMethods(acqStatic)
        model =cgm2.CGM2_3()
        model.configure(detectedCalibrationMethods=dcm)
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)
        model.setStaticTrackingMarkers(actual_trackingMarkers)

        # filtre 1 ModelCalibrationFilter => calibration initiale du model  
        scp = modelFilters.StaticCalibrationProcedure(model)
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model).compute()

        # raffinement des centres articulaires 
        modelDecorator.HipJointCenterDecorator(model).hara()
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="both")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="both")

        # filtre 2 ModelCalibrationFilter => update de la calibration de maniere a prendre en compte les nouveaux centres de rotation  
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                        markerDiameter=markerDiameter).compute()

        import ipdb; ipdb.set_trace()
        #---------------------------------
        # PERSONALISATION
        
        targetvalue = model.getSegment("Pelvis").getreferential("TF").static.getNode_byLabel("pelvisCentre").getLocal()
        
        
        
        #tf.static.addNode("LHJC_cgm1",LHJC_loc,positionType="Local",desc = "Davis")

        #-------------------------------




        # --- OPENSIM SCALING---

        # fichier de configuration Opensim prédéfini pour le CGM2.3   
        osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"interface\\CGM23\\OsimToCGM.settings")
        markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\markerset\\CGM23-markerset.xml"
        osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\pycgm2-gait2354_simbody.osim"
        scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\setup\\CGM23_scaleSetup_template.xml"

        # appel de la procedure "generic"
        proc = opensimScalingInterfaceProcedure.ScalingXmlProcedure(data_path,model.mp["Bodymass"],model.mp["Height"])
        proc.setSetupFiles(osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
        proc.prepareStaticTrial_fromBtkAcq( acqStatic, staticFilename[:-4])
        proc.prepareXml()
        
        # appel du filtre permettant de lancer la procedure et de recuperer le fichier osim
        oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
        oisf.run()
        scaledOsim = oisf.getOsim()
        scaledOsimName = oisf.getOsimName()




        # -------------------------- FIN ---------------------------------

        # Nous avons desormais un fichier osim mis a l echelle qu'il faut venir personaliser. 
        # le osim est un fichier xml. J'ai construit une "interface" permettant de lire/ecrire le xml.  

        osimInterface = opensimInterface.osimInterface(data_path, scaledOsimName)
        # si tu veux recuperer la liste des muscles alors tu peux appeller la methode
        muscles = osimInterface.getMuscles()

        # osimInterface est une instance qui contient un attribut xml
        osimInterface.xml

        # en python , la librairie permettant de lire des fichier xml est BeautifulSoup. 
        #si tu fais 
        osimInterface.xml.m_soup # tu auras tous le contenu du fichier xml

        # voir comment acceder a l 'info que tu veux modifier

        # modifie le m_soup !!!! TOddddddddddddddo

        # une fois que tu as modifié ce que tu veux il faut ensuite faire

        osimInterface.xml.update()





        


        # -------------- Le reste ci dessous : on verra plus tard----------------------
        # procAnaDriven = opensimAnalysesInterfaceProcedure.AnalysesXmlCgmDrivenModelProcedure(DATA_PATH,scaledOsimName,"musculoskeletal_modelling/pose_standstill","CGM2.3")
        # procAnaDriven.setPose("standstill")
        # procAnaDriven.prepareXml()
        # oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAnaDriven)
        # oiamf.run()

        # muscleLengths = opensimIO.OpensimDataFrame(DATA_PATH, "musculoskeletal_modelling/driven_standstill/Driven-CGM23-analyses_MuscleAnalysis_Length.sto")
        # print(muscleLengths.getDataFrame()["rect_fem_r"])
        
        
        # import numpy as np
        # coordinates = scaledOsim.getCoordinateSet()
        # coordinates.get("hip_flexion_r").set_default_value(np.deg2rad(90))
        # coordinates.get("knee_flexion_r").set_default_value(np.deg2rad(-90))

        # states = scaledOsim.initSystem()
        # muscles = scaledOsim.getMuscles()    
        # rf = muscles.get("rect_fem_r")
        # print(rf.getLength(states))
        # ipdb.set_trace()