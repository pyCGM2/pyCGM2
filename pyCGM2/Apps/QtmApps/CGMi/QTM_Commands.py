## coding: utf-8

import argparse

from pyCGM2.Apps.QtmApps.CGMi import CGM1_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM11_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM21_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM22_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM23_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM24_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM25_workflow
from pyCGM2.Apps.QtmApps.CGMi import CGM26_workflow

def main():
    

    parser = argparse.ArgumentParser(prog='QTM-pyCGM2 Operations')


    # create sub-parser
    sub_parsers = parser.add_subparsers(help='',dest="Type")    
    
    # ------ CGM1 ------------ 
    parser_CGM1 = sub_parsers.add_parser('CGM1', help='CGM1 workflow')
    # level 0
    parser_CGM1.add_argument('--sessionFile', type=str,
                        help='setting xml file from qtm', default="session.xml")
    parser_CGM1.add_argument('-ae', '--anomalyException',
                        action='store_true', help='raise an exception if an anomaly is detected')

    # ------ CGM1 ------------
    parser_CGM11 = sub_parsers.add_parser('CGM1.1', help='CGM11 workflow')
    parser_CGM11.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser_CGM11.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')

    # ------ CG21 ------------
    parser_CGM21 = sub_parsers.add_parser('CGM2.1', help='CGM21 workflow')
    parser_CGM21.add_sargument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser_CGM21.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')

    # ------ CG22 ------------
    parser_CGM22 = sub_parsers.add_parser('CGM2.2', help='CGM22 workflow')
    parser_CGM22.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser_CGM22.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM22.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')

    # ------ CG23 ------------
    parser_CGM23 = sub_parsers.add_parser('CGM2.3', help='CGM22 workflow')
    parser_CGM23.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser_CGM23.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser_CGM23.add_argument('-msm','--musculoSkeletalModel', action='store_true', help='musculoskeletal model')

    # ------ CGM24 ------------
    parser_CGM24 = sub_parsers.add_parser('CGM2.4', help='CGM24 workflow')
    parser_CGM24.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser_CGM24.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')

    # ------ CGM25 ------------
    parser_CGM25 = sub_parsers.add_parser('CGM2.5', help='CGM25 workflow')
    parser_CGM25.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser_CGM25.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')

    # ------ CGM26 ------------
    parser_CGM26 = sub_parsers.add_parser('CGM2.6', help='CGM26 workflow')
    parser_CGM26.add_argument('--sessionFile', type=str, help='setting xml file from qtm', default="session.xml")
    parser_CGM26.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')


    args = parser.parse_args()
    # print("\nLes arguments enregistrés sont les suivants : ")
    print(args)

    if args.Type == "CGM1":
        CGM1_workflow.main(args.sessionFile,anomalyException=args.anomalyException)

    elif args.Type == "CGM1.1":
        CGM11_workflow.main(args.sessionFile,anomalyException=args.anomalyException)

    elif args.Type == "CGM2.1":
        CGM21_workflow.main(args.sessionFile,anomalyException=args.anomalyException)

    elif args.Type == "CGM2.2":
        CGM22_workflow.main(args.sessionFile,anomalyException=args.anomalyException,
                            musculoSkeletalModel=args.musculoSkeletalModel)

    elif args.Type == "CGM2.3":
        CGM23_workflow.main(args.sessionFile,anomalyException=args.anomalyException,
                            musculoSkeletalModel=args.musculoSkeletalModel)

    elif args.Type == "CGM2.4":
        CGM24_workflow.main(args.sessionFile,anomalyException=args.anomalyException)

    elif args.Type == "CGM2.5":
        CGM25_workflow.main(args.sessionFile,anomalyException=args.anomalyException)

    elif args.Type == "CGM2.6":
        CGM26_workflow.main(args.sessionFile,anomalyException=args.anomalyException)



    else:
        raise Exception ("[pyCGM2] - command not known. (select: CGM1 CGM1.1 CGM2.1 CGM2.2 CGM2.3 CGM2.4 CGM2.5 CGM2.6)")




if __name__ == '__main__':
    main()