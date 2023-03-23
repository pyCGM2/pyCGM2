## coding: utf-8

import argparse

from pyCGM2.Apps.ViconApps.Events import zeniDetector


def main():
    

    parser = argparse.ArgumentParser(prog='Nexus-pyCGM2 Events Detector')


    # create sub-parser
    sub_parsers = parser.add_subparsers(help='',dest="Type")    
    
    # ------ ZENI ------------
    # 
    parser_zeni = sub_parsers.add_parser('Zeni', help='Zeni event detector')
    # level 0
    parser_zeni.add_argument('-fso', '--footStrikeOffset', type=int,
                        help='systenatic foot strike offset on both side')
    parser_zeni.add_argument('-foo', '--footOffOffset', type=int,
                        help='systenatic foot off offset on both side')


   
    args = parser.parse_args()
    print(args)


    if args.Type == "Zeni":
        zeniDetector.main(args)
    else:
        raise Exception ("[pyCGM2] - command not known. (select: Zeni)")




if __name__ == '__main__':
    main()