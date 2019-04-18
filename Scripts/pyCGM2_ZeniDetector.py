# -*- coding: utf-8 -*-
#import ipdb
import os
import argparse
import traceback
import logging

import pyCGM2
from pyCGM2.Tools import btkTools
from pyCGM2.Events import events
from pyCGM2 import log; log.setLogger()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):
    DATA_PATH = os.getcwd()+"\\"

    file = args.file[0]
    if not os.path.isfile(file):
        raise Exception("the file (%s) doesn t exist" %(file))

    modelledTrials  = [file]

    for trial in modelledTrials:
        logging.info("[pyCGM2]: Zeni Event Detection on trial %s"%(str(trial)))
        acqGait = btkTools.smartReader(str(DATA_PATH + trial))

        acqGait.ClearEvents()
        # ----------------------EVENT DETECTOR-------------------------------
        evp = events.ZeniProcedure()

        if args.footStrikeOffset is not None:
            evp.setFootStrikeOffset(args.footStrikeOffset)
        if args.footOffOffset is not None:
            evp.setFootOffOffset(args.footOffOffset)

        # event filter
        evf = events.EventFilter(evp,acqGait)
        evf.detect()

        btkTools.smartWriter( acqGait, str(DATA_PATH + trial))

        logging.info("[pyCGM2]: Zeni Event Detection on trial %s ----> Done"%(str(trial)))

        if args.MokkaCheck:
            cmd = "Mokka.exe \"%s\""%(str(DATA_PATH + trial))
            os.system(cmd)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ZeniDetector')
    parser.add_argument('file', nargs=1, help='your c3d file')
    parser.add_argument('-fso','--footStrikeOffset', type=int, help='systenatic foot strike offset on both side')
    parser.add_argument('-foo','--footOffOffset', type=int, help='systenatic foot off offset on both side')
    parser.add_argument('--MokkaCheck', action='store_false', help=' Mokka Checking' )
    args = parser.parse_args()

    #---- main script -----
    try:
        main(args)

    except Exception, errormsg:
        print "Script errored!"
        print "Error message: %s" % errormsg
        traceback.print_exc()
        print "Press return to exit.."
        raw_input()
