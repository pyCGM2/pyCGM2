## coding: utf-8

import argparse

from pyCGM2.Apps.ViconApps.CGM1 import CGM1_Calibration, CGM1_Fitting
from pyCGM2.Apps.ViconApps.CGM1_1 import CGM1_1_Calibration, CGM1_1_Fitting
from pyCGM2.Apps.ViconApps.CGM2_1 import CGM2_1_Calibration, CGM2_1_Fitting
from pyCGM2.Apps.ViconApps.CGM2_2 import CGM2_2_Calibration, CGM2_2_Fitting
from pyCGM2.Apps.ViconApps.CGM2_3 import CGM2_3_Calibration, CGM2_3_Fitting
from pyCGM2.Apps.ViconApps.CGM2_4 import CGM2_4_Calibration, CGM2_4_Fitting
from pyCGM2.Apps.ViconApps.CGM2_5 import CGM2_5_Calibration, CGM2_5_Fitting
from pyCGM2.Apps.ViconApps.CGM2_6 import CGM_Knee2DofCalibration, CGM_KneeSARA


def main():
    

    parser = argparse.ArgumentParser(prog='Nexus-pyCGM2 CGM2 Operations')


    # create sub-parser
    sub_parsers = parser.add_subparsers(help='',dest="Type")    
    
    # ------ CGM1 ------------
    # 
    parser_CGM1 = sub_parsers.add_parser('CGM1', help='CGM1 operations')
    # level 0
    CGM1_sub_parsers = parser_CGM1.add_subparsers(help='CGM1 sub-commands',dest="Command")
    # level 0.0
    parser_CGM1_calibration = CGM1_sub_parsers.add_parser('Calibration', help='CGM1 calibration')
    parser_CGM1_calibration.add_argument('-l', '--leftFlatFoot', type=int,
                        help='left flat foot option')
    parser_CGM1_calibration.add_argument('-r', '--rightFlatFoot', type=int,
                        help='right flat foot option')
    parser_CGM1_calibration.add_argument('-hf', '--headFlat', type=int,
                        help='head flat option')
    parser_CGM1_calibration.add_argument('-md', '--markerDiameter',
                        type=float, help='marker diameter')
    parser_CGM1_calibration.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of the model outputs')
    parser_CGM1_calibration.add_argument('--check', action='store_true',
                        help='force cgm1 as model ouput suffix')
    parser_CGM1_calibration.add_argument('--resetMP', action='store_true',
                        help='reset optional anthropometric parameters')
    parser_CGM1_calibration.add_argument('--forceMP', action='store_true',
                        help='force the use of MP offsets to compute knee and ankle joint centres')
    parser_CGM1_calibration.add_argument('-ae', '--anomalyException',
                        action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM1_calibration.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)
    # level 0.1
    parser_CGM1_fitting = CGM1_sub_parsers.add_parser('Fitting', help='CGM1 fitting')
    parser_CGM1_fitting.add_argument(
        '--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser_CGM1_fitting.add_argument('-md', '--markerDiameter',
                        type=float, help='marker diameter')
    parser_CGM1_fitting.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')
    parser_CGM1_fitting.add_argument('--check', action='store_true',
                        help='force cgm1 as model ouput suffix')
    parser_CGM1_fitting.add_argument('-ae', '--anomalyException',
                        action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM1_fitting.add_argument('-fi', '--frameInit', type=int,
                        help='first frame to process')
    parser_CGM1_fitting.add_argument('-fe', '--frameEnd', type=int,
                        help='last frame to process')
    parser_CGM1_fitting.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)


    # ------ CGM11 ------------
    # 
    parser_CGM11 = sub_parsers.add_parser('CGM1.1', help='CGM1.1 operations')
    # level 0
    CGM11_sub_parsers = parser_CGM11.add_subparsers(help='CGM1.1 sub-commands',dest="Command")
    # level 0.0
    parser_CGM11_calibration = CGM11_sub_parsers.add_parser('Calibration', help='CGM1.1 calibration')
    parser_CGM11_calibration.add_argument('-l', '--leftFlatFoot', type=int,
                        help='left flat foot option')
    parser_CGM11_calibration.add_argument('-r', '--rightFlatFoot', type=int,
                        help='right flat foot option')
    parser_CGM11_calibration.add_argument('-hf', '--headFlat', type=int,
                        help='head flat option')
    parser_CGM11_calibration.add_argument('-md', '--markerDiameter',
                        type=float, help='marker diameter')
    parser_CGM11_calibration.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')
    parser_CGM11_calibration.add_argument('--check', action='store_true',
                        help='force ggm1.1 as model output suffix')
    parser_CGM11_calibration.add_argument('--resetMP', action='store_true',
                        help='reset optional anthropometric parameters')
    parser_CGM11_calibration.add_argument('--forceMP', action='store_true',
                        help='force the use of MP offsets to compute knee and ankle joint centres')
    parser_CGM11_calibration.add_argument('-ae', '--anomalyException',
                        action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM11_calibration.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)
    
     # level 0.1
    parser_CGM11_fitting = CGM11_sub_parsers.add_parser('Fitting', help='CGM1.1 fitting')
    parser_CGM11_fitting.add_argument(
        '--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser_CGM11_fitting.add_argument('-md', '--markerDiameter',
                        type=float, help='marker diameter')
    parser_CGM11_fitting.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')
    parser_CGM11_fitting.add_argument('--check', action='store_true',
                        help='force model output suffix')
    parser_CGM11_fitting.add_argument('-ae', '--anomalyException',
                        action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM11_fitting.add_argument('-fi', '--frameInit', type=int,
                        help='first frame to process')
    parser_CGM11_fitting.add_argument('-fe', '--frameEnd', type=int,
                        help='last frame to process')
    parser_CGM11_fitting.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)

    # ------ CGM21 ------------
    # 
    parser_CGM21 = sub_parsers.add_parser('CGM2.1', help='CGM2.1 operations')
    # level 0
    CGM21_sub_parsers = parser_CGM21.add_subparsers(help='CGM2.1 sub-commands',dest="Command")
    # level 0.0
    parser_CGM21_calibration = CGM21_sub_parsers.add_parser('Calibration', help='CGM2.1 calibration')
    parser_CGM21_calibration.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser_CGM21_calibration.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser_CGM21_calibration.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser_CGM21_calibration.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM21_calibration.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM21_calibration.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM21_calibration.add_argument('--resetMP', action='store_true', help='reset optional anthropometric parameters')
    parser_CGM21_calibration.add_argument('--forceMP', action='store_true',
                        help='force the use of MP offsets to compute knee and ankle joint centres')
    parser_CGM21_calibration.add_argument('--forceLHJC', nargs='+')
    parser_CGM21_calibration.add_argument('--forceRHJC', nargs='+')
    parser_CGM21_calibration.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM21_calibration.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)
     # level 0.1
    parser_CGM21_fitting = CGM21_sub_parsers.add_parser('Fitting', help='CGM2.1 fitting')
    parser_CGM21_fitting.add_argument('--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser_CGM21_fitting.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM21_fitting.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM21_fitting.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM21_fitting.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM21_fitting.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser_CGM21_fitting.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')
    parser_CGM21_fitting.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)


    # ------ CGM22 ------------
    # 
    parser_CGM22 = sub_parsers.add_parser('CGM2.2', help='CGM2.2 operations')
    # level 0
    CGM22_sub_parsers = parser_CGM22.add_subparsers(help='CGM2.2 sub-commands',dest="Command")
    # level 0.0
    parser_CGM22_calibration = CGM22_sub_parsers.add_parser('Calibration', help='CGM2.2 calibration')
    parser_CGM22_calibration.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser_CGM22_calibration.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser_CGM22_calibration.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser_CGM22_calibration.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM22_calibration.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM22_calibration.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM22_calibration.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser_CGM22_calibration.add_argument('--forceLHJC', nargs='+')
    parser_CGM22_calibration.add_argument('--forceRHJC', nargs='+')
    parser_CGM22_calibration.add_argument('--resetMP', action='store_true', help='reset optional anthropometric parameters')
    parser_CGM22_calibration.add_argument('--forceMP', action='store_true',
                        help='force the use of MP offsets to compute knee and ankle joint centres')
    parser_CGM22_calibration.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM22_calibration.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')
    parser_CGM22_calibration.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)
    
     # level 0.1
    parser_CGM22_fitting = CGM22_sub_parsers.add_parser('Fitting', help='CGM2.2 fitting')
    parser_CGM22_fitting.add_argument('--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser_CGM22_fitting.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM22_fitting.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM22_fitting.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM22_fitting.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
    parser_CGM22_fitting.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM22_fitting.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser_CGM22_fitting.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')
    parser_CGM22_fitting.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')
    parser_CGM22_fitting.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)
    # ------ CGM23 ------------
    # 
    parser_CGM23 = sub_parsers.add_parser('CGM2.3', help='CGM2.3 operations')
    # level 0
    CGM23_sub_parsers = parser_CGM23.add_subparsers(help='CGM2.3 sub-commands',dest="Command")
    # level 0.0
    parser_CGM23_calibration = CGM23_sub_parsers.add_parser('Calibration', help='CGM2.3 calibration')
    parser_CGM23_calibration.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser_CGM23_calibration.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser_CGM23_calibration.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser_CGM23_calibration.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM23_calibration.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM23_calibration.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM23_calibration.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser_CGM23_calibration.add_argument('--resetMP', action='store_true', help='reset optional anthropometric parameters')
    parser_CGM23_calibration.add_argument('--forceMP', action='store_true',
                        help='force the use of MP offsets to compute knee and ankle joint centres')
    parser_CGM23_calibration.add_argument('--forceLHJC', nargs='+')
    parser_CGM23_calibration.add_argument('--forceRHJC', nargs='+')
    parser_CGM23_calibration.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM23_calibration.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')
    parser_CGM23_calibration.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)
    
     # level 0.1
    parser_CGM23_fitting = CGM23_sub_parsers.add_parser('Fitting', help='CGM2.3 fitting')
    parser_CGM23_fitting.add_argument('--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser_CGM23_fitting.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM23_fitting.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser_CGM23_fitting.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM23_fitting.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM23_fitting.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
    parser_CGM23_fitting.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM23_fitting.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser_CGM23_fitting.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')
    parser_CGM23_fitting.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')
    parser_CGM23_fitting.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)



    # ------ CGM24 ------------
    # 
    parser_CGM24 = sub_parsers.add_parser('CGM2.4', help='CGM2.4 operations')
    # level 0
    CGM24_sub_parsers = parser_CGM24.add_subparsers(help='CGM2.4 sub-commands',dest="Command")
    # level 0.0
    parser_CGM24_calibration = CGM24_sub_parsers.add_parser('Calibration', help='CGM2.4 calibration')
    parser_CGM24_calibration.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser_CGM24_calibration.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser_CGM24_calibration.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser_CGM24_calibration.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM24_calibration.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM24_calibration.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM24_calibration.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser_CGM24_calibration.add_argument('--resetMP', action='store_true', help='reset optional anthropometric parameters')
    parser_CGM24_calibration.add_argument('--forceMP', action='store_true',
                        help='force the use of MP offsets to compute knee and ankle joint centres')
    parser_CGM24_calibration.add_argument('--forceLHJC', nargs='+')
    parser_CGM24_calibration.add_argument('--forceRHJC', nargs='+')
    parser_CGM24_calibration.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM24_calibration.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)
    
     # level 0.1
    parser_CGM24_fitting = CGM24_sub_parsers.add_parser('Fitting', help='CGM2.4 fitting')
    parser_CGM24_fitting.add_argument('--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser_CGM24_fitting.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM24_fitting.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM24_fitting.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM24_fitting.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser_CGM24_fitting.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
    parser_CGM24_fitting.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM24_fitting.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser_CGM24_fitting.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')
    parser_CGM24_fitting.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)


    # ------ CGM25 ------------
    # 
    parser_CGM25 = sub_parsers.add_parser('CGM2.5', help='CGM2.5 operations')
    # level 0
    CGM25_sub_parsers = parser_CGM25.add_subparsers(help='CGM2.5 sub-commands',dest="Command")
    # level 0.0
    parser_CGM25_calibration = CGM25_sub_parsers.add_parser('Calibration', help='CGM2.5 calibration')
    parser_CGM25_calibration.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser_CGM25_calibration.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser_CGM25_calibration.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser_CGM25_calibration.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM25_calibration.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM25_calibration.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM25_calibration.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser_CGM25_calibration.add_argument('--resetMP', action='store_true', help='reset optional anthropometric parameters')
    parser_CGM25_calibration.add_argument('--forceMP', action='store_true',
                        help='force the use of MP offsets to compute knee and ankle joint centres')
    parser_CGM25_calibration.add_argument('--forceLHJC', nargs='+')
    parser_CGM25_calibration.add_argument('--forceRHJC', nargs='+')
    parser_CGM25_calibration.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM25_calibration.add_argument('--offline', nargs=2, help=' subject name and static c3d file', required=False)
    
     # level 0.1
    parser_CGM25_fitting = CGM25_sub_parsers.add_parser('Fitting', help='CGM2.5 fitting')
    parser_CGM25_fitting.add_argument('--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser_CGM25_fitting.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser_CGM25_fitting.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser_CGM25_fitting.add_argument('--check', action='store_true', help='force model output suffix')
    parser_CGM25_fitting.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser_CGM25_fitting.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
    parser_CGM25_fitting.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM25_fitting.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser_CGM25_fitting.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')
    parser_CGM25_fitting.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)


    # ------ CGM26 ------------
    # 
    parser_CGM26 = sub_parsers.add_parser('CGM2.6', help='CGM2.6 operations')
    # level 0
    CGM26_sub_parsers = parser_CGM26.add_subparsers(help='CGM2.6 sub-commands',dest="Command")
    # level 0.0
    parser_CGM26_calibration2dof = CGM26_sub_parsers.add_parser('2DOF', help='2DOF knee functional calibration')
    parser_CGM26_calibration2dof.add_argument('-s','--side', type=str, help="Side : Left or Right")
    parser_CGM26_calibration2dof.add_argument('-b','--beginFrame', type=int, help="begin frame")
    parser_CGM26_calibration2dof.add_argument('-e','--endFrame', type=int, help="end frame")
     # level 0.1
    parser_CGM26_sara = CGM26_sub_parsers.add_parser('SARA', help='SARA knee functional calibration')
    parser_CGM26_sara.add_argument('-s','--side', type=str, help="Side : Left or Right")
    parser_CGM26_sara.add_argument('-b','--beginFrame', type=int, help="begin frame")
    parser_CGM26_sara.add_argument('-e','--endFrame', type=int, help="end frame")
   
    args = parser.parse_args()
    # print("\nLes arguments enregistrés sont les suivants : ")
    print(args)

    if args.Type == "CGM1":
        if args.Command == "Calibration":
            CGM1_Calibration.main(args)
        elif args.Command == "Fitting":
            CGM1_Fitting.main(args)

        else:
            raise Exception ("[pyCGM2] - CGM1 command not known (select: Calibration or Fitting)")
    elif args.Type == "CGM1.1":
        if args.Command == "Calibration":
            CGM1_1_Calibration.main(args)
        elif args.Command == "Fitting":
            CGM1_1_Fitting.main(args)

        else:
            raise Exception ("[pyCGM2] - CGM1_1 command not known (select: Calibration or Fitting)")

    elif args.Type == "CGM2.1":
        if args.Command == "Calibration":
            CGM2_1_Calibration.main(args)
        elif args.Command == "Fitting":
            CGM2_1_Fitting.main(args)

        else:
            raise Exception ("[pyCGM2] - CGM2_1 command not known (select: Calibration or Fitting)")

    elif args.Type == "CGM2.2":
        if args.Command == "Calibration":
            CGM2_2_Calibration.main(args)
        elif args.Command == "Fitting":
            CGM2_2_Fitting.main(args)

        else:
            raise Exception ("[pyCGM2] - CGM2_2 command not known (select: Calibration or Fitting)")

    elif args.Type == "CGM2.3":
        if args.Command == "Calibration":
            CGM2_3_Calibration.main(args)
        elif args.Command == "Fitting":
            CGM2_3_Fitting.main(args)

        else:
            raise Exception ("[pyCGM2] - CGM2_3 command not known (select: Calibration or Fitting)")

    elif args.Type == "CGM2.4":
        if args.Command == "Calibration":
            CGM2_4_Calibration.main(args)
        elif args.Command == "Fitting":
            CGM2_4_Fitting.main(args)
        else:
            raise Exception ("[pyCGM2] - CGM2_4 command not known (select: Calibration or Fitting)")

    elif args.Type == "CGM2.5":
        if args.Command == "Calibration":
            CGM2_5_Calibration.main(args)
        elif args.Command == "Fitting":
            CGM2_5_Fitting.main(args)
        else:
            raise Exception ("[pyCGM2] - CGM2_5 command not known (select: Calibration or Fitting)")

    elif args.Type == "CGM2.6":
        if args.Command == "2DOF":
            CGM_Knee2DofCalibration.main(args)
        elif args.Command == "SARA":
            CGM_KneeSARA.main(args)
        else:
            raise Exception ("[pyCGM2] - CGM2_6 Knee calibration command not known (select: 2DOF or SARA)")


    else:
        raise Exception ("[pyCGM2] - command not known. (select: CGM1 CGM1.1 CGM2.1 CGM2.2 CGM2.3 CGM2.4 CGM2.5 CGM2.6)")




if __name__ == '__main__':
    main()