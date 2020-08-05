# -*- coding: utf-8 -*-
"""Nexus Operation : **plotNormalizedEmg**

The script displays gait-normalized emg envelops

:param -bpf, --BandpassFrequencies [array]: bandpass frequencies
:param -ecf, --EnvelopLowpassFrequency [double]: cut-off low pass frequency for getting emg envelop
:param -c, --consistency [bool]: display consistency plot ( ie : all gait cycles) instead of a descriptive statistics view

Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -bpf 20 450 -ecf=8.9 --consistency
    (bandpass frequencies set to 20 and 450Hz and envelop made from a low-pass filter with a cutoff frequency of 8.9Hz,
    all gait cycles will be displayed)


"""
import os
import logging
import argparse

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)
from pyCGM2.Utils import files
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import nexusFilters,nexusTools

from pyCGM2.Configurator import EmgManager
import ViconNexus


def main():

    parser = argparse.ArgumentParser(description='EMG-plot_temporalEMG')
    parser.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser.add_argument('-elf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser.add_argument('-c','--consistency', action='store_true', help='consistency plots')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED: # run Operation


        # --- acquisition file and path----
        DATA_PATH, inputFileNoExt = NEXUS.GetTrialName()
        inputFile = inputFileNoExt+".c3d"

        #--------------------------settings-------------------------------------
        if os.path.isfile(DATA_PATH + "emg.settings"):
            emgSettings = files.openFile(DATA_PATH,"emg.settings")
            logging.warning("[pyCGM2]: emg.settings detected in the data folder")
        else:
            emgSettings = None

        manager = EmgManager.EmgConfigManager(None,localInternalSettings=emgSettings)
        manager.contruct()


        # ----------------------INPUTS-------------------------------------------
        bandPassFilterFrequencies = manager.BandpassFrequencies#emgSettings["Processing"]["BandpassFrequencies"]
        if args.BandpassFrequencies is not None:
            if len(args.BandpassFrequencies) != 2:
                raise Exception("[pyCGM2] - bad configuration of the bandpass frequencies ... set 2 frequencies only")
            else:
                bandPassFilterFrequencies = [float(args.BandpassFrequencies[0]),float(args.BandpassFrequencies[1])]
                logging.info("Band pass frequency set to %i - %i instead of 20-200Hz",bandPassFilterFrequencies[0],bandPassFilterFrequencies[1])

        envelopCutOffFrequency = manager.EnvelopLowpassFrequency#emgSettings["Processing"]["EnvelopLowpassFrequency"]
        if args.EnvelopLowpassFrequency is not None:
            envelopCutOffFrequency =  args.EnvelopLowpassFrequency
            logging.info("Cut-off frequency set to %i instead of 6Hz ",envelopCutOffFrequency)

        consistencyFlag = True if args.consistency else False


        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,inputFileNoExt,subject)
        acq = nacf.build()


        # --------------emg Processing--------------
        EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT,NORMAL_ACTIVITIES  =  manager.getEmgConfiguration()

        analysis.processEMG_fromBtkAcq(acq, EMG_LABELS,
            highPassFrequencies=bandPassFilterFrequencies,
            envelopFrequency=envelopCutOffFrequency) # high pass then low pass for all c3ds

        emgAnalysis = analysis.makeEmgAnalysis(DATA_PATH, [inputFile], EMG_LABELS,btkAcqs = [acq])


        if not consistencyFlag:
            plot.plotDescriptiveEnvelopEMGpanel(DATA_PATH,emgAnalysis, EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT, NORMAL_ACTIVITIES, normalized=False,exportPdf=True,outputName=inputFile)
        else:
            plot.plotConsistencyEnvelopEMGpanel(DATA_PATH,emgAnalysis, EMG_LABELS,EMG_MUSCLES,EMG_CONTEXT, NORMAL_ACTIVITIES, normalized=False,exportPdf=True,outputName=inputFile)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    main()
