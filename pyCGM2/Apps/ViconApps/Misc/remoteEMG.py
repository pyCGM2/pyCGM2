import argparse
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusTools
from pyCGM2.Gap import gapFillingProcedures
from pyCGM2.Gap import gapFilters
from viconnexusapi import ViconNexus
from pyCGM2.Tools import btkTools
from pyCGM2.EMG import emgFilters

import matplotlib.pyplot as plt

import pyCGM2
LOGGER = pyCGM2.LOGGER


def main(args=None):
    
    # if args  is None:
    parser = argparse.ArgumentParser(description='remote EMG')
    args = parser.parse_args()


    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if NEXUS_PYTHON_CONNECTED:  # run Operation

        DATA_PATH, filenameLabelledNoExt = NEXUS.GetTrialName()

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.logger.info("file: " + filenameLabelledNoExt)

        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("subject %s" % (subject))

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,
            DATA_PATH, filenameLabelledNoExt, subject)
        acq = nacf.build()


        acqEMG =  btkTools.smartReader(DATA_PATH+"marche 11.c3d")

        values16_0 = acq.GetAnalog("Voltage.EMG16").GetValues()
        nframes = values16_0.shape[0]


        values16 = acqEMG.GetAnalog("Emg_16").GetValues()[0:-1:2,:]
        values16 = values16[0:nframes,:]


        bf = emgFilters.BasicEmgProcessingFilter(acq, ["Voltage.EMG16"])
        bf.setHighPassFrequencies(20, 200)
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acq,["Voltage.EMG16"])
        envf.setCutoffFrequency(6.0)
        envf.run()

        bf = emgFilters.BasicEmgProcessingFilter(acqEMG, ["Emg_16"])
        bf.setHighPassFrequencies(20, 200)
        bf.run()

        envf = emgFilters.EmgEnvelopProcessingFilter(acqEMG,["Emg_16"])
        envf.setCutoffFrequency(6.0)
        envf.run()

        btkTools.smartWriter(acq,"test.c3d")
        values16_0_Env = acq.GetAnalog("Voltage.EMG16_Rectify_Env").GetValues()

        values16_Env = acqEMG.GetAnalog("Emg_16_Rectify_Env").GetValues()[0:-1:2,:]
        values16_Env = values16_Env[0:nframes,:]



        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('difference picoEMG intra TP ( numerique(bleu) vs analogique(rouge))')
        ax1.plot(values16_0[4800:8420,:],"-r")
        ax1.plot(values16[4800:8420,:]/1000,"-b")

        ax2.plot(values16_0_Env[4800:8420,:],"-r")
        ax2.plot(values16_Env[4800:8420,:]/1000,"-b")

        ax1.set_title("raw")
        ax2.set_title("rectify (20-200Hz) - env(6Hz)")
        plt.show()


        import ipdb; ipdb.set_trace()
    else:
        return 0

if __name__ == "__main__":

    main()
