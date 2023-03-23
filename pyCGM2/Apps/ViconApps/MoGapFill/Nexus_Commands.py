## coding: utf-8

import argparse

from pyCGM2.Apps.ViconApps.MoGapFill import KalmanGapFilling


def main():
    

    parser = argparse.ArgumentParser(prog='Nexus-pyCGM2 gap filling')


    # create sub-parser
    sub_parsers = parser.add_subparsers(help='',dest="Type")    
    
    # ------ ZENI ------------
    # 
    parser_kalman = sub_parsers.add_parser('Kalman', help='Kalman gap filling')
    # level 0
    


   
    args = parser.parse_args()
    print(args)


    if args.Type == "Kalman":
        KalmanGapFilling.main(args)
    else:
        raise Exception ("[pyCGM2] - command not known. (select: Kalman)")




if __name__ == '__main__':
    main()