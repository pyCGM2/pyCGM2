import pyCGM2
from pyCGM2.Lib import plot
from pyCGM2.Lib import emg
from pyCGM2.Lib import analysis

def example1():

    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\Session 1 - CGM1\\"

    trialNames = ["03367_05136_20200604-GBNNN-VDEF-01.c3d"]

    emgManager = emg.loadEmg(DATA_PATH)

    emg.processEMG(DATA_PATH, trialNames, emgManager.getChannels(),
        highPassFrequencies=[20,200],
        envelopFrequency=6.0)

    figs = plot.plotTemporalEMG(DATA_PATH, trialNames[0],
            rectify = True)


def example2():

    DATA_PATH = pyCGM2.TEST_DATA_PATH + "GaitData\\Patient\\Session 1 - CGM1\\"

    trialNames = ["03367_05136_20200604-GBNNN-VDEF-01.c3d", "03367_05136_20200604-GBNNN-VDEF-02.c3d"]

    emgManager = emg.loadEmg(DATA_PATH)

    emg.processEMG(DATA_PATH, trialNames, emgManager.getChannels(),
        highPassFrequencies=[20,200],
        envelopFrequency=6.0)

    analysisInstance = analysis.makeAnalysis(DATA_PATH,
                        trialNames,
                        emgChannels = emgManager.getChannels()                )

    plot.plotDescriptiveEnvelopEMGpanel(DATA_PATH,analysisInstance)



if __name__ == '__main__':
    example1()
