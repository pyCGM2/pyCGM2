## coding: utf-8

import argparse


from pyCGM2.Apps.Commands import initSettingsCmd

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

from pyCGM2.Apps.QtmApps.CGMi import CGM1_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM11_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM21_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM22_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM23_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM24_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM25_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM26_workflow


class NEXUS_PlotsParser(object):
    def __init__(self,nexusSubparser):
        self.nexusSubparser = nexusSubparser

    def constructParsers(self):
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
    def __init__(self,nexusSubparser,cgmVersion):
        self.nexusSubparser = nexusSubparser
        self.cgmVersion = cgmVersion
        self.cgmVersionShort = self.cgmVersion.replace(".","")


    def constructParsers(self):
        cgm_parsers = self.nexusSubparser.add_parser(self.cgmVersion, help= f"{self.cgmVersion} commands")
        cgm_subparsers = cgm_parsers.add_subparsers(help='', dest=self.cgmVersionShort)

        calibrationparser = self.setCalibrationParser(cgm_subparsers)
        fittingParser = self.setFittingParser(cgm_subparsers)



    def __calibrationParser(self,calibrationParser):

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

    def __fittingParser(self,fittingParser):
        
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

    def setCalibrationParser(self,cgm_subparsers):

        calibrationParser = cgm_subparsers.add_parser('Calibration', 
                                                      help=f"Calibration command of the {self.cgmVersion}")
        calibrationParser = self.__calibrationParser(calibrationParser)

        return  calibrationParser  

    def setFittingParser(self,cgm_subparsers):
        fittingParser = cgm_subparsers.add_parser('Fitting', 
                                                  help=f"Fitting command of the {self.cgmVersion}")
        fittingParser = self.__fittingParser(fittingParser)          

        return fittingParser

class QTM_CGMparser(object):
    def __init__(self,qtmSubparser,cgmVersion):
        self.qtmSubparser = qtmSubparser
        self.cgmVersion = cgmVersion
        self.cgmVersionShort = self.cgmVersion.replace(".","")


    def constructParsers(self):

        cgm_parsers = self.qtmSubparser.add_parser(self.cgmVersion, help= f"{self.cgmVersion} commands")

        cgm_parsers.add_argument('--sessionFile', type=str,
                            help='setting xml file from qtm', default="session.xml")
        cgm_parsers.add_argument('-ae', '--anomalyException',
                            action='store_true', help='raise an exception if an anomaly is detected')

        if self.cgmVersion in ["CGM2.2","CGM2.3"]:
           cgm_parsers.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')



class MainParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers(help='sub-command help', dest='subparser')

        self.Global()
        self.Nexus()
        self.QTM()

    def Settings(self):

        nexusparser = self.subparsers.add_parser('SETTINGS', help='pyCGM2 settings')
        nexus_subparser = nexusparser.add_subparsers(help='', dest='SETTINGS')

        # folder init
        parser_init = nexus_subparser.add_parser("Edit", help= "folder initialisation commands")
        parser_init.add_argument('-m', '--model', type=str,  help='copy CGM settings')
        parser_init.add_argument('-e', '--emg', action='store_true',  help='copy emg settings')



    def Nexus(self):

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

        qtmparser = self.subparsers.add_parser('QTM', help='Vicon nexus commands')
        qtm_subparser = qtmparser.add_subparsers(help='', dest='QTM')

        QTM_CGMparser(qtm_subparser,"CGM1.0").constructParsers()
        QTM_CGMparser(qtm_subparser,"CGM1.1").constructParsers()
        QTM_CGMparser(qtm_subparser,"CGM2.1").constructParsers()
        QTM_CGMparser(qtm_subparser,"CGM2.2").constructParsers()
        QTM_CGMparser(qtm_subparser,"CGM2.3").constructParsers()
        QTM_CGMparser(qtm_subparser,"CGM2.4").constructParsers()
        QTM_CGMparser(qtm_subparser,"CGM2.5").constructParsers()
        QTM_CGMparser(qtm_subparser,"CGM2.6").constructParsers()

    def run(self,debug=False):
        
        args = self.parser.parse_args()
        print(args)

        if not debug:
            if "SETTINGS" in args:
                if args.SETTINGS == "Edit":
                    initSettingsCmd.main(args)

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
                        if args.kinetics == "Temporal":
                            kinetics.temporal(args)
                        elif args.kinetics == "Normalized":
                            kinetics.normalized(args)
                        elif args.kinetics == "Comparison":
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
                if args.QTM == "CGM1.0":
                    CGM1_workflow.main(args)
                elif args.QTM == "CGM1.0":
                    CGM11_workflow.main(args)
                elif args.QTM == "CGM2.1":
                    CGM21_workflow.main(args)
                elif args.QTM == "CGM2.2":
                    CGM22_workflow.main(args)
                elif args.QTM == "CGM2.3":
                    CGM23_workflow.main(args)
                elif args.QTM == "CGM2.4":
                    CGM24_workflow.main(args)
                elif args.QTM == "CGM2.5":
                    CGM25_workflow.main(args)
                elif args.QTM == "CGM2.6":
                    CGM26_workflow.main(args)


def main():
    my_class = MainParser()
    my_class.run(debug=False) 



if __name__ == '__main__':
    main()




