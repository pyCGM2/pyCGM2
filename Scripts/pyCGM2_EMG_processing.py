# -*- coding: utf-8 -*-
#import ipdb
import os
import argparse
import traceback
import logging

import pyCGM2
from pyCGM2.Utils import files
from pyCGM2.Configurator import EmgManager
from pyCGM2 import log; log.setLogger()
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):
    DATA_PATH = os.getcwd()+"\\"

    # User Settings
    if os.path.isfile(DATA_PATH + args.userFile):
        userSettings = files.openFile(DATA_PATH,args.userFile)
    else:
        raise Exception ("user setting file not found")

    # internal (expert) Settings
    if args.expertFile:
        if os.path.isfile(DATA_PATH + args.expertFile):
            internalSettings = files.openFile(DATA_PATH,args.expertFile)
        else:
            raise Exception ("expert setting file not found")
    else:
        internalSettings = None


    # --- Manager ----
    manager = EmgManager.EmgConfigManager(userSettings,localInternalSettings=internalSettings)
    manager.contruct()
    finalSettings = manager.getFinalSettings()
    files.prettyDictPrint(finalSettings)

    # reconfiguration of emg settings as lists
    EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT,NORMAL_ACTIVITIES  =  manager.getEmgConfiguration()
    

    #----- rectified view -------
    rectTrials = manager.temporal_trials
    analysis.processEMG(DATA_PATH, rectTrials, EMG_LABELS,
        highPassFrequencies=manager.BandpassFrequencies,
        envelopFrequency=manager.EnvelopLowpassFrequency,fileSuffix=None) # high pass then low pass for all c3ds

    for trial in rectTrials:
        plot.plotTemporalEMG(DATA_PATH,trial, EMG_LABELS,EMG_MUSCLES, EMG_CONTEXT, NORMAL_ACTIVITIES,exportPdf=True,rectify=manager.rectifyFlag)


    #----- Gait normalized envelop -------
    envTrials = manager.gaitNormalized_trials
    analysis.processEMG(DATA_PATH, envTrials, EMG_LABELS,
        highPassFrequencies=manager.BandpassFrequencies,
        envelopFrequency=manager.EnvelopLowpassFrequency,fileSuffix=None) # high pass then low pass for all c3ds

    emgAnalysisInstance = analysis.makeEmgAnalysis(DATA_PATH, envTrials, EMG_LABELS,None, None)

    filename = manager.title

    if not manager.consistencyFlag:
        plot.plotDescriptiveEnvelopEMGpanel(DATA_PATH,emgAnalysisInstance, EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT, NORMAL_ACTIVITIES, normalized=False,exportPdf=True,outputName=filename)
    else:
        plot.plotConsistencyEnvelopEMGpanel(DATA_PATH,emgAnalysisInstance, EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT, NORMAL_ACTIVITIES, normalized=False,exportPdf=True,outputName=filename)


    logging.info("=============Writing of final Settings=============")
    files.saveJson(DATA_PATH, str(filename+"-EMG.completeSettings"), finalSettings)
    logging.info("---->complete settings (%s) exported" %(str(filename +"-EMG.completeSettings")))


    raw_input("Press return to exit..")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='EMG-pipeline')
    parser.add_argument('--userFile', type=str, help='userSettings', default="emg.userSettings")
    parser.add_argument('--expertFile', type=str, help='Local expert settings')
    args = parser.parse_args()
        #print args

        # ---- main script -----
    try:
        main(args)


    except Exception, errormsg:
        print "Script errored!"
        print "Error message: %s" % errormsg
        traceback.print_exc()
        print "Press return to exit.."
        raw_input()
        #
