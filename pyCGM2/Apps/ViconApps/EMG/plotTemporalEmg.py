# -*- coding: utf-8 -*-
"""Nexus Operation : **plotTemporalEmg**

The script displays rectified EMG with time as x-axis

:param -bpf, --BandpassFrequencies [array]: bandpass frequencies
:param -ecf, --EnvelopLowpassFrequency [double]: cut-off low pass frequency for getting emg envelop
:param -r, --raw [bool]: display non-rectified emg instead of rectified

Examples:
    In the script argument box of a python nexus operation, you can edit:

    >>>  -bpf 20 450 -ecf=8.9 --raw
    (bandpass frequencies set to 20 and 450Hz and envelop made from a low-pass filter with a cutoff frequency of 8.9Hz,
    non-rectified EMG  will be displayed)


"""
import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse


import pyCGM2

from pyCGM2.Utils import files
from pyCGM2.Lib import analysis
from pyCGM2.Lib import plot
from pyCGM2.Lib import emg
from pyCGM2.Report import normativeDatasets

from pyCGM2.Nexus import nexusFilters,nexusTools
from pyCGM2.Configurator import EmgManager
from viconnexusapi import ViconNexus


def main():

    parser = argparse.ArgumentParser(description='EMG-plot_temporalEMG')
    parser.add_argument('-bpf', '--BandpassFrequencies', nargs='+',help='bandpass filter')
    parser.add_argument('-ecf','--EnvelopLowpassFrequency', type=int, help='cutoff frequency for emg envelops')
    parser.add_argument('-r','--raw', action='store_true', help='rectified data')
    parser.add_argument('-ina','--ignoreNormalActivity', action='store_true', help='do not display normal activity')
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
            LOGGER.logger.warning("[pyCGM2]: emg.settings detected in the data folder")
        else:
            emgSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"emg.settings")

        # ----------------------INPUTS-------------------------------------------
        bandPassFilterFrequencies = emgSettings["Processing"]["BandpassFrequencies"]
        if args.BandpassFrequencies is not None:
            if len(args.BandpassFrequencies) != 2:
                raise Exception("[pyCGM2] - bad configuration of the bandpass frequencies ... set 2 frequencies only")
            else:
                bandPassFilterFrequencies = [float(args.BandpassFrequencies[0]),float(args.BandpassFrequencies[1])]
                LOGGER.logger.info("Band pass frequency set to %i - %i instead of 20-200Hz",bandPassFilterFrequencies[0],bandPassFilterFrequencies[1])

        envelopCutOffFrequency = emgSettings["Processing"]["EnvelopLowpassFrequency"]
        if args.EnvelopLowpassFrequency is not None:
            envelopCutOffFrequency =  args.EnvelopLowpassFrequency
            LOGGER.logger.info("Cut-off frequency set to %i instead of 6Hz ",envelopCutOffFrequency)

        rectifyBool = False if args.raw else True

        # --------------------------SUBJECT ------------------------------------
        subject = nexusTools.getActiveSubject(NEXUS)


        # btk Acquisition
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,inputFileNoExt,subject)
        acq = nacf.build()

        emgChannels = list()
        for channel in emgSettings["CHANNELS"].keys():
            if emgSettings["CHANNELS"][channel]["Muscle"] is not None and emgSettings["CHANNELS"][channel]["Muscle"] != "None" :
                emgChannels.append(channel)

        emg.processEMG_fromBtkAcq(acq, emgChannels,
            highPassFrequencies=bandPassFilterFrequencies,
            envelopFrequency=envelopCutOffFrequency) # high pass then low pass for all c3ds

        plot.plotTemporalEMG(DATA_PATH,inputFile, emgSettings,exportPdf=True,rectify=rectifyBool,
                            btkAcq=acq,ignoreNormalActivity= args.ignoreNormalActivity)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":


    main()
