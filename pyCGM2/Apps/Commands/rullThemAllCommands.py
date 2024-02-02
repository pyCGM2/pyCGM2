## coding: utf-8
"""
This module provides a command-line interface for managing and executing various tasks in the pyCGM2 framework.
It includes commands for setting up and running different versions of the Conventional Gait Analysis Model (CGM),
as well as generating plots, handling events, and performing gap filling within Vicon Nexus and Qualisys Track Manager (QTM) environments.

Users can leverage this interface to streamline their workflow in gait analysis and biomechanical studies using pyCGM2.
"""

import os
import argparse

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files

from pyCGM2.Apps.ViconApps.CGM1 import CGM1_Calibration, CGM1_Fitting
from pyCGM2.Apps.ViconApps.CGM1_1 import CGM1_1_Calibration, CGM1_1_Fitting
from pyCGM2.Apps.ViconApps.CGM2_1 import CGM2_1_Calibration, CGM2_1_Fitting
from pyCGM2.Apps.ViconApps.CGM2_2 import CGM2_2_Calibration, CGM2_2_Fitting
from pyCGM2.Apps.ViconApps.CGM2_3 import CGM2_3_Calibration, CGM2_3_Fitting
from pyCGM2.Apps.ViconApps.CGM2_4 import CGM2_4_Calibration, CGM2_4_Fitting
from pyCGM2.Apps.ViconApps.CGM2_5 import CGM2_5_Calibration, CGM2_5_Fitting
from pyCGM2.Apps.ViconApps.CGM2_6 import CGM_Knee2DofCalibration, CGM_KneeSARA
from pyCGM2.Apps.ViconApps.commands import deviceDetailsCommand

from pyCGM2.Apps.ViconApps.Events import zeniDetector
from pyCGM2.Apps.ViconApps.MoGapFill import KalmanGapFilling
from pyCGM2.Apps.ViconApps.MoGapFill import GloersenGapFilling

from pyCGM2.Apps.ViconApps.Plot import spatioTemporalParameters
from pyCGM2.Apps.ViconApps.Plot import kinematics
from pyCGM2.Apps.ViconApps.Plot import scores
from pyCGM2.Apps.ViconApps.Plot import kinetics
from pyCGM2.Apps.ViconApps.Plot import emg

from pyCGM2.Apps.QtmApps.CGMi import QPYCGM2_events
from pyCGM2.Apps.QtmApps.CGMi import QPYCGM2_modelling
from pyCGM2.Apps.QtmApps.CGMi import QPYCGM2_processing








class NEXUS_PlotsParser(object):
    """
    Responsible for parsing and managing plot-related commands within the Vicon Nexus environment.
    This class facilitates the creation and organization of sub-parsers for various plot types, 
    such as spatio-temporal parameters (STP), kinematics, kinetics, and EMG. These sub-parsers 
    are integrated into a higher-level command parser, enabling structured and efficient handling 
    of plot commands.

    Args:
        nexusSubparser (argparse.ArgumentParser): A higher-level command parser for integrating plot sub-commands.

    Sub-Parsers and their Arguments:
        'STP' (Spatiotemporal Parameters) Parser:
            - '-ps', '--pointSuffix': Suffix added to model outputs, type: str.

        'Kinematics' Parser:
            - 'Temporal':
                - '-ps', '--pointSuffix': Suffix of model outputs, type: str.
            - 'Normalized':
                - '-nd', '--normativeData': Normative data set (e.g., 'Schwartz2008'), type: str.
                - '-ndm', '--normativeDataModality': Normative data modality (varies based on dataset), type: str.
                - '-ps', '--pointSuffix': Suffix of model outputs, type: str.
                - '-c', '--consistency': Consistency plots, action: 'store_true'.
            - 'Comparison':
                - Similar to Normalized parser.
            - 'MAP':
                - Similar to Normalized parser, excluding the 'consistency' option.

        'Kinetics' Parser:
            - 'Temporal':
                - '-ps', '--pointSuffix': Suffix of model outputs, type: str.
            - 'Normalized':
                - '-nd', '--normativeData': Normative data set, type: str.
                - '-ndm', '--normativeDataModality': Normative data modality, type: str.
                - '-ps', '--pointSuffix': Suffix of model outputs, type: str.
                - '-c', '--consistency': Consistency plots, action: 'store_true'.
            - 'Comparison':
                - Similar to Normalized parser.

        'EMG' Parser:
            - 'Temporal':
                - '-bpf', '--BandpassFrequencies': Bandpass filter frequencies, nargs: '+', type: list.
                - '-elf', '--EnvelopLowpassFrequency': Cutoff frequency for EMG envelops, type: int.
                - '-r', '--raw': Non-rectified data, action: 'store_true'.
                - '-ina', '--ignoreNormalActivity': Do not display normal activity, action: 'store_true'.
            - 'Normalized':
                - Similar to Temporal parser, excluding the 'raw' and 'ignoreNormalActivity' options.
                - '-c', '--consistency': Consistency plots, action: 'store_true'.
            - 'Comparison':
                - Similar to Normalized parser.
    """
    def __init__(self, nexusSubparser: argparse.ArgumentParser):
        self.nexusSubparser = nexusSubparser

    def constructParsers(self):
        """
        Constructs and adds sub-parsers for different plot categories within the Nexus parser.
        
        Each parser and sub-parser is configured with specific arguments and options relevant to the respective plot category. 
        The parsers are designed to capture and process command line arguments for generating various types of plots in the Nexus environment.
        """
        # plot--------------
        plot_parser = self.nexusSubparser.add_parser("Plots", help= "Plot commands")
        plot_subparsers = plot_parser.add_subparsers(help='', dest="Plots")

        # ------ STP ------------
        parser_stp = plot_subparsers.add_parser('STP', help='SpatiotemporalParameters plots')
        parser_stp.add_argument('-ps','--pointSuffix', type=str, help='suffix added to model outputs')


        # ------ KINEMATICS ------------
        parser_kinematics = plot_subparsers.add_parser('Kinematics', help='Kinematics plots')
            
        kinematics_sub_parsers = parser_kinematics.add_subparsers(help='kinematics plots sub-commands',dest="Kinematics")

        # temporal
        parser_kinematicsTemporal = kinematics_sub_parsers.add_parser('Temporal', help='temporal plot')
        parser_kinematicsTemporal.add_argument('-ps', '--pointSuffix', type=str,
                            help='suffix of model outputs')
        # Normalized
        parser_kinematicsNormalized = kinematics_sub_parsers.add_parser('Normalized', help='time-normalized')
        parser_kinematicsNormalized.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
        parser_kinematicsNormalized.add_argument('-ndm','--normativeDataModality', type=str,
                            help="if Schwartz2008 [VerySlow,Slow,Free,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                            default="Free")
        parser_kinematicsNormalized.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
        parser_kinematicsNormalized.add_argument('-c','--consistency', action='store_true', help='consistency plots')
        # comparison
        parser_kinematicsComparison = kinematics_sub_parsers.add_parser('Comparison', help='time-normalized comparison')
        parser_kinematicsComparison.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
        parser_kinematicsComparison.add_argument('-ndm','--normativeDataModality', type=str,
                            help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                            default="Free")
        parser_kinematicsComparison.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
        parser_kinematicsComparison.add_argument('-c','--consistency', action='store_true', help='consistency plots')

        #MAP
        parser_kinematicsMAP = kinematics_sub_parsers.add_parser('MAP', help='Mouvement analysis profile')
        parser_kinematicsMAP.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
        parser_kinematicsMAP.add_argument('-ndm','--normativeDataModality', type=str,
                            help="if Schwartz2008 [VerySlow,Slow,Free,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                            default="Free")
        parser_kinematicsMAP.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')

        # ------ KINETICS ------------

        parser_kinetics = plot_subparsers.add_parser('Kinetics', help='kinetics plots')
            
        kinetics_sub_parsers = parser_kinetics.add_subparsers(help='kinetics plots sub-commands',dest="Kinetics")

        # level 0.0
        parser_kineticsTemporal = kinetics_sub_parsers.add_parser('Temporal', help='temporal plot')
        parser_kineticsTemporal.add_argument('-ps', '--pointSuffix', type=str,
                            help='suffix of model outputs')

        # level 0.1
        parser_kineticsNormalized = kinetics_sub_parsers.add_parser('Normalized', help='time-normalized')
        parser_kineticsNormalized.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
        parser_kineticsNormalized.add_argument('-ndm','--normativeDataModality', type=str,
                            help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                            default="Free")
        parser_kineticsNormalized.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
        parser_kineticsNormalized.add_argument('-c','--consistency', action='store_true', help='consistency plots')

        # level 0.2
        parser_kineticsComparison = kinetics_sub_parsers.add_parser('Comparison', help='time-normalized comparison')   
        parser_kineticsComparison.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
        parser_kineticsComparison.add_argument('-ndm','--normativeDataModality', type=str,
                            help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                            default="Free")
        parser_kineticsComparison.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
        parser_kineticsComparison.add_argument('-c','--consistency', action='store_true', help='consistency plots')


        # ------ EMG ------------

        parser_emg = plot_subparsers.add_parser('EMG', help='EMG plots')
    
        emg_sub_parsers = parser_emg.add_subparsers(help='Emg plots sub-commands',dest="EMG")


        parser_emgTemporal = emg_sub_parsers.add_parser('Temporal', help='temporal plot')
        parser_emgTemporal.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
        parser_emgTemporal.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
        parser_emgTemporal.add_argument('-r','--raw', action='store_true', help='non rectified data')
        parser_emgTemporal.add_argument('-ina','--ignoreNormalActivity', action='store_true', help='do not display normal activity')


        parser_emgNormalized = emg_sub_parsers.add_parser('Normalized', help='time-normalized')
        parser_emgNormalized.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
        parser_emgNormalized.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
        parser_emgNormalized.add_argument('-c','--consistency', action='store_true', help='consistency plots')


        parser_emgComparison = emg_sub_parsers.add_parser('Comparison', help='time-normalized comparison')
        parser_emgComparison.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
        parser_emgComparison.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
        parser_emgComparison.add_argument('-c','--consistency', action='store_true', help='consistency plots')


class NEXUS_CGMparser(object):
    """
    Handles the parsing of commands specific to different versions of the Conventional Gait Analysis Model (CGM) within the Vicon Nexus environment.
    This class constructs parsers for calibration and fitting commands, with options varying based on the CGM version.

    Args:
        nexusSubparser (argparse.ArgumentParser): A higher-level command parser for integrating CGM-specific sub-commands.
        cgmVersion (str): The version of CGM for which the parser is being constructed.

    'Calibration' Parser Arguments:
        - '-l', '--leftFlatFoot': Option for left flat foot, type: int.
        - '-r', '--rightFlatFoot': Option for right flat foot, type: int.
        - '-hf', '--headFlat': Option for head flat, type: int.
        - '-md', '--markerDiameter': Marker diameter, type: float.
        - '-ps', '--pointSuffix': Suffix of the model outputs, type: str.
        - '--check': Force CGM version as model output suffix, action: 'store_true'.
        - '--resetMP': Reset optional anthropometric parameters, action: 'store_true'.
        - '--forceMP': Force the use of MP offsets for joint centres calculation, action: 'store_true'.
        - '-ae', '--anomalyException': Raise an exception if an anomaly is detected, action: 'store_true'.
        - '--offline': Subject name and static c3d file, nargs: 2, required: False.
        - '--forceLHJC': Forces the left hip joint center, nargs: '+', available in CGM versions 2.1 to 2.5.
        - '--forceRHJC': Forces the right hip joint center, nargs: '+', available in CGM versions 2.1 to 2.5.
        - '-msm','--musculoSkeletalModel': Musculoskeletal model, action: 'store_true', available in CGM versions 2.2 and 2.3.
        - '--noIk': Cancel inverse kinematic, action: 'store_true', available in CGM versions 2.3 to 2.5.

    'Fitting' Parser Arguments:
        - '-md', '--markerDiameter': Marker diameter, type: float.
        - '-ps', '--pointSuffix': Suffix of model outputs, type: str.
        - '--check': Force CGM version as model output suffix, action: 'store_true'.
        - '-ae', '--anomalyException': Raise an exception if an anomaly is detected, action: 'store_true'.
        - '-fi', '--frameInit': First frame to process, type: int.
        - '-fe', '--frameEnd': Last frame to process, type: int.
        - '--offline': Subject name, dynamic c3d file, and mfpa, nargs: 3, required: False.
        - '-c3d', '--c3d': Load the c3d file, type: str, to avoid loading data from Nexus API.
        - '--proj': Referential to project joint moment. Options: 'Distal', 'Proximal', 'Global' for CGM1.0; 'JCS', 'Distal', 'Proximal', 'Global' for CGM versions 1.1, 2.1 to 2.5.
        - '-msm','--musculoSkeletalModel': Musculoskeletal model, action: 'store_true', available in CGM versions 2.2 and 2.3.
        - '-a','--accuracy': Inverse Kinematics accuracy, type: float, available in CGM versions 2.2 to 2.5.
        - '--noIk': Cancel inverse kinematic, action: 'store_true', available in CGM versions 2.2 to 2.5.
    """  
    def __init__(self, nexusSubparser: argparse.ArgumentParser, cgmVersion: str):
        self.nexusSubparser = nexusSubparser
        self.cgmVersion = cgmVersion
        self.cgmVersionShort = self.cgmVersion.replace(".","")


    def constructParsers(self):
        """
        Constructs parsers for CGM-specific commands such as calibration and fitting. 
        It adds these parsers under the main CGM version parser.
        """
        cgm_parsers = self.nexusSubparser.add_parser(self.cgmVersion, help= f"{self.cgmVersion} commands")
        cgm_subparsers = cgm_parsers.add_subparsers(help='', dest=self.cgmVersionShort)

        calibrationparser = self.setCalibrationParser(cgm_subparsers)
        fittingParser = self.setFittingParser(cgm_subparsers)



    def __calibrationParser(self, calibrationParser: argparse.ArgumentParser):
        """
        Constructs and configures the calibration parser with arguments specific to the CGM version.

        The common arguments include options for flat foot, head position, marker diameter, etc.
        Additional version-specific arguments are added based on the CGM version.

        Args:
            calibrationParser (argparse.ArgumentParser): The parser to which calibration arguments will be added.

        Returns:
            argparse.ArgumentParser: The updated calibration parser with added arguments.
        """

        calibrationParser.add_argument('-l', '--leftFlatFoot', type=int,
                            help='left flat foot option')
        calibrationParser.add_argument('-r', '--rightFlatFoot', type=int,
                            help='right flat foot option')
        calibrationParser.add_argument('-hf', '--headFlat', type=int,
                            help='head flat option')
        calibrationParser.add_argument('-md', '--markerDiameter',
                            type=float, help='marker diameter')
        calibrationParser.add_argument('-ps', '--pointSuffix', type=str,
                            help='suffix of the model outputs')
        calibrationParser.add_argument('--check', action='store_true',
                            help=f"force {self.cgmVersionShort} as model ouput suffix")
        calibrationParser.add_argument('--resetMP', action='store_true',
                            help='reset optional anthropometric parameters')
        calibrationParser.add_argument('--forceMP', action='store_true',
                            help='force the use of MP offsets to compute knee and ankle joint centres')
        calibrationParser.add_argument('-ae', '--anomalyException',
                            action='store_true', help='raise an exception if an anomaly is detected')
        calibrationParser.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)

        if self.cgmVersion in ["CGM2.1","CGM2.2","CGM2.3","CGM2.4","CGM2.5"]:
            calibrationParser.add_argument('--forceLHJC', nargs='+')
            calibrationParser.add_argument('--forceRHJC', nargs='+')

        if self.cgmVersion in ["CGM2.2","CGM2.3"]:
            calibrationParser.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')

        if self.cgmVersion in ["CGM2.3","CGM2.4","CGM2.5"]:
            calibrationParser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    
        return calibrationParser

    def __fittingParser(self, fittingParser: argparse.ArgumentParser):
        """
        Constructs and configures the fitting parser with arguments specific to the CGM version.

        Common arguments include marker diameter, point suffix, anomaly exception handling, etc.
        Additional version-specific arguments are added based on the CGM version.

        Args:
            fittingParser (argparse.ArgumentParser): The parser to which fitting arguments will be added.

        Returns:
            argparse.ArgumentParser: The updated fitting parser with added arguments.
        """
        
        fittingParser.add_argument('-md', '--markerDiameter',
                            type=float, help='marker diameter')
        fittingParser.add_argument('-ps', '--pointSuffix', type=str,
                            help='suffix of model outputs')
        fittingParser.add_argument('--check', action='store_true',
                            help=f"force {self.cgmVersionShort} as model ouput suffix")        
        fittingParser.add_argument('-ae', '--anomalyException',
                            action='store_true', help='raise an exception if an anomaly is detected')
        fittingParser.add_argument('-fi', '--frameInit', type=int,
                            help='first frame to process')
        fittingParser.add_argument('-fe', '--frameEnd', type=int,
                            help='last frame to process')
        fittingParser.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)
        fittingParser.add_argument('-c3d', '--c3d', type=str,
                            help='load the c3d file. This operation avoid to load data from the Nexus API. \
                            (be sure you save the c3d beforehand  )')


        if self.cgmVersion in ["CGM1.0"]:
            fittingParser.add_argument(
                '--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal or Global')

        if self.cgmVersion in ["CGM1.1","CGM2.1", "CGM2.2","CGM2.3","CGM2.4","CGM2.5"]:
            fittingParser.add_argument(
                '--proj', type=str, help='Referential to project joint moment. Choice : JCS, Distal, Proximal or Global')

        if self.cgmVersion in ["CGM2.2","CGM2.3"]:
            fittingParser.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')

        if self.cgmVersion in ["CGM2.2","CGM2.3","CGM2.4","CGM2.5"]:
            fittingParser.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
            fittingParser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')

        return fittingParser

    def setCalibrationParser(self, cgm_subparsers: argparse._SubParsersAction):
        """
        Sets up the calibration parser as a sub-parser under the main CGM parser.

        Args:
            cgm_subparsers (argparse._SubParsersAction): The subparsers action object to which the calibration parser is to be added.

        Returns:
            argparse.ArgumentParser: The calibration parser configured with CGM-specific arguments.
        """

        calibrationParser = cgm_subparsers.add_parser('Calibration', 
                                                      help=f"Calibration command of the {self.cgmVersion}")
        calibrationParser = self.__calibrationParser(calibrationParser)

        return  calibrationParser  

    def setFittingParser(self, cgm_subparsers: argparse._SubParsersAction):
        """
        Sets up the fitting parser as a sub-parser under the main CGM parser.

        Args:
            cgm_subparsers (argparse._SubParsersAction): The subparsers action object to which the fitting parser is to be added.

        Returns:
            argparse.ArgumentParser: The fitting parser configured with CGM-specific arguments.
        """
        fittingParser = cgm_subparsers.add_parser('Fitting', 
                                                  help=f"Fitting command of the {self.cgmVersion}")
        fittingParser = self.__fittingParser(fittingParser)          

        return fittingParser


class MainParser:
    """
    Main parser class for handling command line arguments for various functionalities within the pyCGM2 framework.
    This class sets up sub-parsers for settings, Vicon Nexus, and Qualisys Track Manager (QTM) related commands.

    The parser organizes the commands into different categories, each responsible for a specific aspect of the CGM analysis workflow.

    Sub-Parsers and their Arguments:
        'SETTINGS':
            - 'Edit': For folder initialization commands.
                - '-m', '--model': Copy CGM settings, type: str.
                - '-e', '--emg': Copy EMG settings, action: 'store_true'.

        'NEXUS':
            - 'CGM1.0' to 'CGM2.5' parsers with their  arguments.
            - 'CGM2.6' Parser:
                - '2DOF': Two Degree of Freedom knee functional calibration.
                    - '-s', '--side': Specify the side (Left or Right), type: str.
                    - '-fi', '--frameInit': First frame to process, type: int.
                    - '-fe', '--frameEnd': Last frame to process, type: int.
                - 'SARA': SARA knee functional calibration.
                    - '-s', '--side': Specify the side (Left or Right), type: str.
                    - '-fi', '--frameInit': First frame to process, type: int.
                    - '-fe', '--frameEnd': Last frame to process, type: int.
            - 'Events' for event-related commands.
                - 'Zeni': Zeni kinematic-based event detection.
                    - '-fso', '--footStrikeOffset': Systematic foot strike offset, type: int.
                    - '-foo', '--footOffOffset': Systematic foot off offset, type: int.
            - 'Gaps' for gap filling commands.
                - 'Kalman': Kalman gap filling, '--markers': list of markers.
                - 'Gloersen': Gloersen gap filling, '--markers': list of markers.
            - 'Plots': For generating various plots.
                
            - 'System': For Nexus system commands.
                - 'DeviceDetails': Command to get device details.

        'QTM':
            - CGM parsers specific to QTM environment.

    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers(help='sub-command help', dest='subparser')

        self.Settings()
        self.Nexus()
        self.QTM()

    def Settings(self):
        """
        Sets up the parser for SETTINGS related commands.

        Creates a sub-parser for SETTINGS with options for editing folder initialization and EMG settings.
        """

        nexusparser = self.subparsers.add_parser('SETTINGS', help='pyCGM2 settings')
        nexus_subparser = nexusparser.add_subparsers(help='', dest='SETTINGS')

        # folder init
        parser_init = nexus_subparser.add_parser("Edit", help= "folder initialisation commands")
        parser_init.add_argument('-m', '--model', type=str,  help='copy CGM settings')
        parser_init.add_argument('-e', '--emg', action='store_true',  help='copy emg settings')



    def Nexus(self):
        """
        Sets up the parser for NEXUS related commands.

        Creates sub-parsers for different CGM versions, events, gap filling, plots, and system commands within the Vicon Nexus environment.
        """
        # First level subparser
        nexusparser = self.subparsers.add_parser('NEXUS', help='Vicon nexus commands')
        nexus_subparser = nexusparser.add_subparsers(help='', dest='NEXUS')

        # Nexus CGM        
        NEXUS_CGMparser(nexus_subparser,"CGM1.0").constructParsers()
        NEXUS_CGMparser(nexus_subparser,"CGM1.1").constructParsers()
        NEXUS_CGMparser(nexus_subparser,"CGM2.1").constructParsers()
        NEXUS_CGMparser(nexus_subparser,"CGM2.2").constructParsers()
        NEXUS_CGMparser(nexus_subparser,"CGM2.3").constructParsers()
        NEXUS_CGMparser(nexus_subparser,"CGM2.4").constructParsers()
        NEXUS_CGMparser(nexus_subparser,"CGM2.5").constructParsers()

        cgm26_parser = nexus_subparser.add_parser("CGM2.6", help= "CGM2.6 commands")
        cgm26_subparsers = cgm26_parser.add_subparsers(help='', dest="CGM26")
        parser_CGM26_calibration2dof = cgm26_subparsers.add_parser('2DOF', help='2DOF knee functional calibration')
        parser_CGM26_calibration2dof.add_argument('-s','--side', type=str, help="Side : Left or Right")
        parser_CGM26_calibration2dof.add_argument('-fi', '--frameInit', type=int,
                            help='first frame to process')
        parser_CGM26_calibration2dof.add_argument('-fe', '--frameEnd', type=int,
                            help='last frame to process')


        parser_CGM26_sara = cgm26_subparsers.add_parser('SARA', help='SARA knee functional calibration')
        parser_CGM26_sara.add_argument('-s','--side', type=str, help="Side : Left or Right")
        parser_CGM26_sara.add_argument('-fi', '--frameInit', type=int,
                            help='first frame to process')
        parser_CGM26_sara.add_argument('-fe', '--frameEnd', type=int,
                            help='last frame to process')



        # events--------------
        event_parser = nexus_subparser.add_parser("Events", help= "events commands")
        event_subparsers = event_parser.add_subparsers(help='', dest="Events")
        parser_zeni = event_subparsers.add_parser('Zeni', help='zeni kinematic-based event detection')
        parser_zeni.add_argument('-fso', '--footStrikeOffset', type=int,
                        help='systenatic foot strike offset on both side')
        parser_zeni.add_argument('-foo', '--footOffOffset', type=int,
                        help='systenatic foot off offset on both side')


        # gapFill--------------
        gap_parser = nexus_subparser.add_parser("Gaps", help= "Gap filling commands")
        gap_subparsers = gap_parser.add_subparsers(help='', dest="Gaps")
        parser_kalman = gap_subparsers.add_parser('Kalman', help='kalman gap filling')
        parser_kalman.add_argument('--markers', nargs='*', help='list of markers',required=False)
        
        parser_gloersen = gap_subparsers.add_parser('Gloersen', help='Gloersen gap filling')
        parser_gloersen.add_argument('--markers', nargs='*', help='list of markers',required=False)
        

        # plot--------------

        NEXUS_PlotsParser(nexus_subparser).constructParsers()
        
        # system----
        system_parser = nexus_subparser.add_parser("System", help= "Nexus system commands")
        system_subparsers = system_parser.add_subparsers(help='', dest="System")
        parser_deviceDetails = system_subparsers.add_parser('DeviceDetails', help='command to get device details')
        



    def QTM(self):
        """
        Sets up the parser for QTM related commands.
        """

        qtmparser = self.subparsers.add_parser('QTM', help='Vicon nexus commands')
        qtm_subparser = qtmparser.add_subparsers(help='', dest='QTM')

        event_parser = qtm_subparser.add_parser("GaitEvents", help= "GaitEvents commands")

        qtm_cgm_parser = qtm_subparser.add_parser("CGM", help= "CGM commands")
        qtm_cgm_subparser = qtm_cgm_parser.add_subparsers(help='', dest='CGM')

        cgmModelling_parser = qtm_cgm_subparser.add_parser("Modelling", help= "CGM Modelling command")
        cgmModelling_parser.add_argument('--debug', action='store_true',
                            help='set logger as debug mode')
        cgmProcessing_parser = qtm_cgm_subparser.add_parser("Processing", help= "CGM processing command")
        cgmProcessing_parser.add_argument('--debug', action='store_true',
                            help='set logger as debug mode')


    def get_parser(self):
        """
        Returns the argument parser.
        """
        return self.parser
    
    def run(self,debug:bool=False):
        """
        Executes the parser and processes the parsed arguments to trigger appropriate functionalities based on the user's input.

        Args:
            debug (bool): If True, runs the parser in debug mode for testing purposes. Default is False.
        """
        
        args = self.parser.parse_args()
        print(args)

        if not debug:
            if "SETTINGS" in args:
                if args.SETTINGS == "Edit":
                    if args.emg:
                        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                                            os.getcwd()+"\\"+"emg.settings")
                        LOGGER.logger.info("[pyCGM2] file [emg.settings] copied in your data folder")
                        os.startfile(os.getcwd()+"\\"+"emg.settings")

                    if args.model is not None:
                        if args.model == "CGM1" or args.model == "CGM1.0":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER
                                            + "CGM1-pyCGM2.settings", os.getcwd()+"\\"+"CGM1-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM1-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM1-pyCGM2.settings")
                        if args.model == "CGM1.1":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM1_1-pyCGM2.settings",
                                            os.getcwd()+"\\"+"CGM1_1-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM1_1-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM1_1-pyCGM2.settings")
                        elif args.model == "CGM2.1":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_1-pyCGM2.settings",
                                            os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM2_1-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
                        elif args.model == "CGM2.2":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_2-pyCGM2.settings",
                                            os.getcwd()+"\\"+"CGM2_2-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM2_2-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM2_2-pyCGM2.settings")
                        elif args.model == "CGM2.3":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_3-pyCGM2.settings",
                                            os.getcwd()+"\\"+"CGM2_3-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM2_3-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM2_3-pyCGM2.settings")

                        elif args.model == "CGM2.4":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_4-pyCGM2.settings",
                                            os.getcwd()+"\\"+"CGM2_4-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM2_4-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM2_4-pyCGM2.settings")

                        elif args.model == "CGM2.5":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                                            os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM2_5-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")

                        elif args.model == "CGM2.6":
                            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                                            os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")
                            LOGGER.logger.info("[pyCGM2] file [CGM2_5-pyCGM2.settings] copied in your data folder")
                            os.startfile(os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")

                        else:
                            LOGGER.logger.error("[pyCGM2] model version not know (CGM1, CGM1.1 ... CGM2.5)")
                            raise


            elif "NEXUS" in args:

                if args.NEXUS == "CGM1.0":
                    if args.CGM10 == "Calibration":
                        CGM1_Calibration.main(args)
                    if args.CGM10 == "Fitting":
                        CGM1_Fitting.main(args)

                elif args.NEXUS == "CGM1.1":
                    if args.CGM11 == "Calibration":
                        CGM1_1_Calibration.main(args)
                    if args.CGM11 == "Fitting":
                        CGM1_1_Fitting.main(args)


                elif args.NEXUS == "CGM2.1":
                    if args.CGM21 == "Calibration":
                        CGM2_1_Calibration.main(args)
                    if args.CGM21 == "Fitting":
                        CGM2_1_Fitting.main(args)   

                elif args.NEXUS == "CGM2.2":
                    if args.CGM22 == "Calibration":
                        CGM2_2_Calibration.main(args)
                    if args.CGM22 == "Fitting":
                        CGM2_2_Fitting.main(args)

                elif args.NEXUS == "CGM2.3":
                    if args.CGM23 == "Calibration":
                        CGM2_3_Calibration.main(args)
                    if args.CGM23 == "Fitting":
                        CGM2_3_Fitting.main(args)

                elif args.NEXUS == "CGM2.4":
                    if args.CGM24 == "Calibration":
                        CGM2_4_Calibration.main(args)
                    if args.CGM24 == "Fitting":
                        CGM2_4_Fitting.main(args)

                elif args.NEXUS == "CGM2.5":
                    if args.CGM25 == "Calibration":
                        CGM2_5_Calibration.main(args)
                    if args.CGM25 == "Fitting":
                        CGM2_5_Fitting.main(args)


                elif args.NEXUS == "CGM2.6":
                    if args.CGM26 == "SARA":
                        CGM_KneeSARA.main(args)
                    if args.CGM26 == "2DOF":
                        CGM_Knee2DofCalibration.main(args)

                # -- Events---
                elif args.NEXUS == "Events":
                    if args.Events == "Zeni":
                        zeniDetector.main(args)

                # -- Gaps---
                elif args.NEXUS == "Gaps":
                    if args.Gaps == "Kalman":
                        KalmanGapFilling.main(args)
                    if args.Gaps == "Gloersen":
                        GloersenGapFilling.main(args)
                
                #--Plots---
                elif args.NEXUS == "Plots":
                    if args.Plots == "STP":
                        spatioTemporalParameters.horizontalHistogram(args)
                    elif args.Plots == "Kinematics":
                        if args.Kinematics == "Temporal":
                            kinematics.temporal(args)
                        elif args.Kinematics == "Normalized":
                            kinematics.normalized(args)
                        elif args.Kinematics == "Comparison":
                            kinematics.normalizedComparison(args)
                        elif args.Kinematics == "MAP":
                            scores.map(args)

                        
                    elif args.Plots == "Kinetics":
                        if args.Kinetics == "Temporal":
                            kinetics.temporal(args)
                        elif args.Kinetics == "Normalized":
                            kinetics.normalized(args)
                        elif args.Kinetics == "Comparison":
                            kinetics.normalizedComparison(args)

                    elif args.Plots == "EMG":
                        if args.EMG == "Temporal":
                            emg.temporal(args)
                        elif args.EMG == "Normalized":
                            emg.normalized(args)
                        elif args.EMG == "Comparison":
                            emg.normalizedComparison(args)

                elif args.NEXUS == "System":
                    if args.System == "DeviceDetails":
                        deviceDetailsCommand.main(args)

            elif "QTM" in args:
                if args.QTM == "CGM" and args.CGM == "Modelling":
                    QPYCGM2_modelling.main(args)
                elif args.QTM == "CGM" and args.CGM == "Processing":
                    QPYCGM2_processing.main(args)
                elif args.QTM == "GaitEvents":
                    QPYCGM2_events.main(args)    

def get_main_parser():
    """
    This function is used by sphinx-argparse for documentation.
    It returns the main argument parser.
    """
    my_parser = MainParser()
    return my_parser.get_parser()

def main():
    my_class = MainParser()
    my_class.run(debug=False) 



if __name__ == '__main__':
    main()




