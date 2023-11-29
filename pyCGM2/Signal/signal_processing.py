"""
The module only contains functions for processing  data
"""
from scipy import signal
import numpy as np
from scipy.interpolate import interp1d
import pyCGM2
LOGGER = pyCGM2.LOGGER
try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

from typing import List, Tuple, Dict, Optional, Union, Callable

# ---- EMG -----

def remove50hz(array:np.ndarray, fa:float):
    """
    Removes the 50Hz frequency component from the provided signal array.

    Args:
        array (np.ndarray): The input array containing the signal data.
        fa (float): Sampling frequency of the signal.

    Returns:
        np.ndarray: The filtered signal array with the 50Hz component removed.
    """
    bEmgStop, aEMGStop = signal.butter(
        2, np.array([49.9, 50.1]) / ((fa*0.5)), 'bandstop')
    value = signal.filtfilt(bEmgStop, aEMGStop, array, axis=0)

    return value


def highPass(array:np.ndarray, lowerFreq:float, upperFreq:float, fa:float):
    """
    Applies a Butterworth bandpass filter to the input signal.

    Args:
        array (np.ndarray): The input array containing the signal data.
        lowerFreq (float): Lower frequency bound for the bandpass filter.
        upperFreq (float): Upper frequency bound for the bandpass filter.
        fa (float): Sampling frequency of the signal.

    Returns:
        np.ndarray: The bandpass-filtered signal.
    """
    bEmgHighPass, aEmgHighPass = signal.butter(
        2, np.array([lowerFreq, upperFreq]) / ((fa*0.5)), 'bandpass')
    value = signal.filtfilt(bEmgHighPass, aEmgHighPass,
                            array-np.mean(array), axis=0)

    return value


def rectify(array:np.ndarray):
    """
    Rectifies a signal by taking the absolute value of each element.

    Args:
        array (np.ndarray): The input array containing the signal data.

    Returns:
        np.ndarray: The rectified signal.
    """
    return np.abs(array)


def enveloppe(array:np.ndarray, fc:float, fa:float):
    """
    Obtains the envelope of a signal using a low-pass filter.

    Args:
        array (np.ndarray): The input array containing the signal data.
        fc (float): Cut-off frequency for the low-pass filter.
        fa (float): Sampling frequency of the signal.

    Returns:
        np.ndarray: The envelope of the signal.
    """
    bEmgEnv, aEMGEnv = signal.butter(2, fc / (fa*0.5), btype='lowpass')
    value = signal.filtfilt(bEmgEnv, aEMGEnv, array, axis=0)
    return value


# ---- btkAcq -----
def markerFiltering(btkAcq:btk.btkAcquisition, markers:List[str], order:int=2, fc:float=6, zerosFiltering:bool=True):
    """
    Applies low-pass filtering to specified markers in a btkAcquisition object.

    Args:
        btkAcq (btk.btkAcquisition): btk acquisition instance to be filtered.
        markers (List[str]): List of marker names to be filtered.
        order (int, optional): Order of the Butterworth filter. Defaults to 2.
        fc (float, optional): Cut-off frequency for the filter. Defaults to 6 Hz.
        zerosFiltering (bool, optional): Enable filtering out zeros. Defaults to True.

    Returns:
        None: The function modifies the btkAcquisition object in place.
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

        filtValues_section = []
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


def forcePlateFiltering(btkAcq:btk.btkAcquisition, order:int=4, fc:float=5):
    """
    Applies low-pass filtering to force plate outputs in a btkAcquisition object.

    Args:
        btkAcq (btk.btkAcquisition): btk acquisition instance to be filtered.
        order (int, optional): Order of the Butterworth filter. Defaults to 4.
        fc (float, optional): Cut-off frequency for the filter. Defaults to 5 Hz.

    Returns:
        None: The function modifies the btkAcquisition object in place.
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


def arrayLowPassFiltering(valuesArray:np.ndarray, freq:float, order:int=2, fc:float=6):
    """
    Applies a low-pass filter to a numpy array.

    Args:
        valuesArray (np.ndarray): Array of values to be filtered.
        freq (float): Sampling frequency of the array.
        order (int, optional): Order of the Butterworth filter. Defaults to 2.
        fc (float, optional): Cut-off frequency for the filter. Defaults to 6 Hz.

    Returns:
        np.ndarray: The low-pass-filtered array.
    """


    b, a = signal.butter(order, fc / (freq*0.5), btype='lowpass')

    out = np.zeros(valuesArray.shape)
    for i in range(0, valuesArray.shape[1]):
        out[:, i] = signal.filtfilt(b, a, valuesArray[:, i])

    return out




def downsample(array:np.ndarray, initFreq:float, targetedFreq:float):
    """
    Downsampling a signal from an initial frequency to a targeted frequency.

    Args:
        array (np.ndarray): Array of values representing the signal.
        initFreq (float): Initial sampling frequency of the signal.
        targetedFreq (float): Targeted sampling frequency after downsampling.

    Returns:
        np.ndarray: The downsampled signal array.

    Raises:
        ValueError: If the targeted frequency is higher than the initial frequency.
    """
    if targetedFreq >= initFreq:
        raise ValueError("targeted frequency cannot be over the initial frequency")
        
    time = np.linspace(0, (array.shape[0] - 1) / initFreq, array.shape[0])
    newTime = np.linspace(0, time[-1], int(array.shape[0] * targetedFreq / initFreq))

    f = interp1d(time, array, axis=0, fill_value="extrapolate")
    newarray = f(newTime)
    
    return newarray

