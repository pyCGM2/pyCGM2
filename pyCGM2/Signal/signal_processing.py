# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Signal
#APIDOC["Draft"]=False
#--end--

"""
The module only contains functions for filtering data
"""
from scipy import signal
from scipy import integrate
import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")


# ---- EMG -----

def remove50hz(array, fa):
    """
    Remove the 50Hz signal

    Args
        array (array(n,n):  array
        fa (float): sample frequency
   """
    bEmgStop, aEMGStop = signal.butter(
        2, np.array([49.9, 50.1]) / ((fa*0.5)), 'bandstop')
    value = signal.filtfilt(bEmgStop, aEMGStop, array, axis=0)

    return value


def highPass(array, lowerFreq, upperFreq, fa):
    """butterworth bandpass filter.

    Args:
        array (numpy.array(n,n)): array
        lowerFreq (float): lower frequency
        upperFreq (float): upper frequency
        fa (float):  sample frequency
   """
    bEmgHighPass, aEmgHighPass = signal.butter(
        2, np.array([lowerFreq, upperFreq]) / ((fa*0.5)), 'bandpass')
    value = signal.filtfilt(bEmgHighPass, aEmgHighPass,
                            array-np.mean(array), axis=0)

    return value


def rectify(array):
    """
    rectify a signal ( i.e get absolute values)

    Args:
        array (numpy.array(n,n)): array

   """
    return np.abs(array)


def enveloppe(array, fc, fa):
    """
    Get signal enveloppe from a low pass filter

    Args:
        array (numpy.array(n,n)): array
        fc (float): cut-off frequency
        fa (float): sample frequency
   """
    bEmgEnv, aEMGEnv = signal.butter(2, fc / (fa*0.5), btype='lowpass')
    value = signal.filtfilt(bEmgEnv, aEMGEnv, array, axis=0)
    return value


# ---- btkAcq -----
def markerFiltering(btkAcq, markers, order=2, fc=6, zerosFiltering=True):
    """
    Low-pass filtering of all points in an acquisition

    Args:
        btkAcq (btk.Acquisition): btk acquisition instance
        fc (float,Optional): cut-off frequency. Default set to 6 Hz
        order (int,optional): order of the low-pass filter, Default set to 2
        zerosFiltering (bool,optional): enable zero filtering, Default set to True
    """

    def filterZeros(array, b, a):

        N = len(array)
        indexes = list(range(0, N))

        for i in range(0, N):
            if array[i] == 0:
                indexes[i] = -1

        splitdata = [x[x != 0] for x in np.split(
            array, np.where(array == 0)[0]) if len(x[x != 0])]
        splitIndexes = [
            x[x != -1] for x in np.split(indexes, np.where(indexes == -1)[0]) if len(x[x != -1])]

        filtValues_section = list()
        for data in splitdata:
            # default as defined in https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
            padlen = 3 * max(len(a), len(b))
            if len(data) <= padlen:
                padlen = len(data) - 1
            filtValues_section.append(signal.filtfilt(
                b, a, data, padlen=padlen, axis=0))

        indexes = np.concatenate(splitIndexes) if splitIndexes != [] else []
        values = np.concatenate(filtValues_section) if filtValues_section != [] else []

        out = np.zeros((N))
        j = 0
        for i in indexes:
            out[i] = values[j]
            j += 1
        return out

    fp = btkAcq.GetPointFrequency()
    bPoint, aPoint = signal.butter(order, fc / (fp*0.5), btype='lowpass')

    for pointIt in btk.Iterate(btkAcq.GetPoints()):
        if pointIt.GetType() == btk.btkPoint.Marker and pointIt.GetLabel() in markers:
            label = pointIt.GetLabel()
            if zerosFiltering:
                x = filterZeros(pointIt.GetValues()[:, 0], bPoint, aPoint)
                y = filterZeros(pointIt.GetValues()[:, 1], bPoint, aPoint)
                z = filterZeros(pointIt.GetValues()[:, 2], bPoint, aPoint)
            else:
                x = signal.filtfilt(
                    bPoint, aPoint, pointIt.GetValues()[:, 0], axis=0)
                y = signal.filtfilt(
                    bPoint, aPoint, pointIt.GetValues()[:, 1], axis=0)
                z = signal.filtfilt(
                    bPoint, aPoint, pointIt.GetValues()[:, 2], axis=0)

            btkAcq.GetPoint(label).SetValues(np.array([x, y, z]).transpose())

            # pointIt.SetValues(np.array( [x,y,z] ).transpose())


def forcePlateFiltering(btkAcq, order=4, fc=5):
    """
    Low-pass filtering of Force plate outputs

    Args:
        btkAcq (btk.Acquisition): btk acquisition instance
        fc (float,Optional): cut-off frequency. Default set to 5 Hz
        order (int,optional): order of the low-pass filter, Default set to 4
    """

    fp = btkAcq.GetAnalogFrequency()
    bPoint, aPoint = signal.butter(order, fc / (fp*0.5), btype='lowpass')

    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()

    for i in range(0, pfc.GetItemNumber()):

        for j in range(0, pfc.GetItem(i).GetChannelNumber()):

            values = pfc.GetItem(i).GetChannel(j).GetValues()[:, 0]

            values_filt = signal.filtfilt(bPoint, aPoint, values, axis=0)

            # SetValues on channel not store new values
            label = pfc.GetItem(i).GetChannel(j).GetLabel()
            try:
                btkAcq.GetAnalog(label).SetValues(values_filt)
            except RuntimeError:
                LOGGER.logger.error(
                    "[pyCGM2] filtering of the force place %i impossible - label %s not found" % (i, label))

# ----- methods ---------


def arrayLowPassFiltering(valuesArray, freq, order=2, fc=6):
    """
    Low-pass filtering of a numpy.array

    Args:
        valuesArray (numpy.array(n,n)) array
        fc (float,Optional): cut-off frequency. Default set to 6 Hz
        order (int,optional): order of the low-pass filter, Default set to 2
    """


    b, a = signal.butter(order, fc / (freq*0.5), btype='lowpass')

    out = np.zeros(valuesArray.shape)
    for i in range(0, valuesArray.shape[1]):
        out[:, i] = signal.filtfilt(b, a, valuesArray[:, i])

    return out
