## coding: utf-8

import argparse
from pyCGM2.Apps.ViconApps.Plot import kinematics
from pyCGM2.Apps.ViconApps.Plot import scores
from pyCGM2.Apps.ViconApps.Plot import kinetics
from pyCGM2.Apps.ViconApps.Plot import emg


def main():
    

    parser = argparse.ArgumentParser(prog='pyCGM2-Nexus Plotting')


    # create sub-parser
    sub_parsers = parser.add_subparsers(help='',dest="Type")    
    
    # # ------ template ------------
    #  parser_kinematics = sub_parsers.add_parser('Kinematics', help='Kinematics plots')
         
    # # level 0 - 
    # kinematics_sub_parsers = parser_kinematics.add_subparsers(help='kinematics plots sub-commands',dest="command")
    # kinematics_sub_parsers.add_argument(.....)
    # ....

    # # level 0.0 - temporal
    # parser_kinematicsTemporal = kinematics_sub_parsers.add_parser('temporal', help='temporal plot')
    # parser_kinematicsTemporal.add_argument(.....)
    # .....
    
    # # level 0.1 - normalized
    # parser_kinematicsNormalized = kinematics_sub_parsers.add_parser('normalized', help='time-normalized')
    # parser_kinematicsNormalized.add_argument()

    # # level 0.2 - compare
    # parser_kinematicsComparison = kinematics_sub_parsers.add_parser('compare', help='time-normalized comparison')
    # parser_kinematicsComparison.add_argument(.....)
    # ....
    # # ------ END template ------------
    

    # ------ KINEMATICS ------------
    # 
    parser_kinematics = sub_parsers.add_parser('Kinematics', help='Kinematics plots')
         
    # level 0
    kinematics_sub_parsers = parser_kinematics.add_subparsers(help='kinematics plots sub-commands',dest="Command")

    # level 0.0
    parser_kinematicsTemporal = kinematics_sub_parsers.add_parser('temporal', help='temporal plot')
    parser_kinematicsTemporal.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')
    # level 0.1
    parser_kinematicsNormalized = kinematics_sub_parsers.add_parser('normalized', help='time-normalized')
    parser_kinematicsNormalized.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser_kinematicsNormalized.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser_kinematicsNormalized.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_kinematicsNormalized.add_argument('-c','--consistency', action='store_true', help='consistency plots')
    # level 0.2
    parser_kinematicsComparison = kinematics_sub_parsers.add_parser('compare', help='time-normalized comparison')
    parser_kinematicsComparison.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser_kinematicsComparison.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser_kinematicsComparison.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_kinematicsComparison.add_argument('-c','--consistency', action='store_true', help='consistency plots')

    # level 0.3
    parser_kinematicsMAP = kinematics_sub_parsers.add_parser('map', help='Mouvement analysis profile')

    parser_kinematicsMAP.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser_kinematicsMAP.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser_kinematicsMAP.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')

    
    # ------ KINETICS ------------
    # 
    parser_kinetics = sub_parsers.add_parser('kinetics', help='kinetics plots')
         
    # level 0
    kinetics_sub_parsers = parser_kinetics.add_subparsers(help='kinetics plots sub-commands',dest="Command")

    # level 0.0
    parser_kineticsTemporal = kinetics_sub_parsers.add_parser('temporal', help='temporal plot')
    parser_kineticsTemporal.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')

    # level 0.1
    parser_kineticsNormalized = kinetics_sub_parsers.add_parser('normalized', help='time-normalized')
    parser_kineticsNormalized.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser_kineticsNormalized.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser_kineticsNormalized.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_kineticsNormalized.add_argument('-c','--consistency', action='store_true', help='consistency plots')

    # level 0.2
    parser_kineticsComparison = kinetics_sub_parsers.add_parser('compare', help='time-normalized comparison')   
    parser_kineticsComparison.add_argument('-nd','--normativeData', type=str, help='normative Data set (Schwartz2008 or Pinzone2014)', default="Schwartz2008")
    parser_kineticsComparison.add_argument('-ndm','--normativeDataModality', type=str,
                        help="if Schwartz2008 [VerySlow,SlowFree,Fast,VeryFast] - if Pinzone2014 [CentreOne,CentreTwo]",
                        default="Free")
    parser_kineticsComparison.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_kineticsComparison.add_argument('-c','--consistency', action='store_true', help='consistency plots')


    # ------ EMG ------------

    parser_emg = sub_parsers.add_parser('EMG', help='EMG plots')
   
   # level 0
    emg_sub_parsers = parser_emg.add_subparsers(help='Emg plots sub-commands',dest="Command")


    # level 0.0
    parser_emgTemporal = emg_sub_parsers.add_parser('temporal', help='temporal plot')
    parser_emgTemporal.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser_emgTemporal.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser_emgTemporal.add_argument('-r','--raw', action='store_true', help='non rectified data')
    parser_emgTemporal.add_argument('-ina','--ignoreNormalActivity', action='store_true', help='do not display normal activity')


     # level 0.1
    parser_emgNormalized = emg_sub_parsers.add_parser('normalized', help='time-normalized')
    parser_emgNormalized.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser_emgNormalized.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser_emgNormalized.add_argument('-c','--consistency', action='store_true', help='consistency plots')

    # level 0.2
    parser_emgComparison = emg_sub_parsers.add_parser('compare', help='time-normalized comparison')
    parser_emgComparison.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser_emgComparison.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser_emgComparison.add_argument('-c','--consistency', action='store_true', help='consistency plots')


   
    args = parser.parse_args()
    # print("\nLes arguments enregistrés sont les suivants : ")
    print(args)

    if args.Type == "Kinematics":
        if args.Command == "temporal":
            kinematics.temporal(args)
        elif args.Command == "normalized":
            kinematics.normalized(args)
        elif args.Command == "compare":
            kinematics.normalizedComparison(args)
        elif args.Command == "map":
            scores.map(args)
        else:
            raise Exception ("[pyCGM2] - Kinematics command not known")


    elif args.Type == "Kinetics":
        if args.Command == "temporal":
            kinetics.temporal(args)
        elif args.Command == "normalized":
            kinetics.normalized(args)
        elif args.Command == "compare":
            kinetics.normalizedComparison(args)
        else:
            raise Exception ("[pyCGM2] - Kinetics command not known")

    elif args.Type == "EMG":
        if args.Command == "temporal":
            emg.temporal(args)
        elif args.Command == "normalized":
            emg.normalized(args)
        elif args.Command == "compare":
            emg.normalizedComparison(args)
        else:
            raise Exception ("[pyCGM2] - EMG command not known")
    else:
        raise Exception ("[pyCGM2] - command not known. choose Kinematics - Kinetics or EMG")




if __name__ == '__main__':
    main()