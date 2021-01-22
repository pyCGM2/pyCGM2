# -*- coding: utf-8 -*-
# from __future__ import print_function
import logging
import numpy as np
from scipy import signal, integrate
try: 
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk



# ---- EMG -----

def remove50hz(array,fa):
    """
        Remove 50Hz signal

        :Parameters:
            - `array` (numpy.array(n,n)) - array
            - `fa` (double) - sample frequency
   """
    bEmgStop, aEMGStop = signal.butter(2, np.array([49.9, 50.1]) / ((fa*0.5)), 'bandstop')
    value= signal.filtfilt(bEmgStop, aEMGStop, array,axis=0  )

    return value

def highPass(array,lowerFreq,upperFreq,fa):
    """
        High pass filtering

        :Parameters:
            - `array` (numpy.array(n,n)) - array
            - `lowerFreq` (double) - lower frequency
            - `upperFreq` (double) - upper frequency
            - `fa` (double) - sample frequency
   """
    bEmgHighPass, aEmgHighPass = signal.butter(2, np.array([lowerFreq, upperFreq]) / ((fa*0.5)), 'bandpass')
    value = signal.filtfilt(bEmgHighPass, aEmgHighPass,array-np.mean(array),axis=0  )

    return value

def rectify(array):
    """
        rectify a signal ( i.e get absolute values)

        :Parameters:
            - `array` (numpy.array(n,n)) - array

   """
    return np.abs(array)

def enveloppe(array, fc,fa):
    """
        Get signal enveloppe from a low pass filter

        :Parameters:
            - `array` (numpy.array(n,n)) - array
            - `fc` (double) - cut-off frequency
            - `fa` (double) - sample frequency
   """
    bEmgEnv, aEMGEnv = signal.butter(2, fc / (fa*0.5) , btype='lowpass')
    value = signal.filtfilt(bEmgEnv, aEMGEnv, array ,axis=0  )
    return value




# ---- btkAcq -----
def markerFiltering(btkAcq,markers,order=2, fc=6,zerosFiltering=True):

    """
        Low-pass filtering of all points in an acquisition

        :Parameters:
            - `btkAcq` (btkAcquisition) - btk acquisition instance
            - `fc` (double) - cut-off frequency
            - `order` (double) - order of the low-pass filter
    """

    def filterZeros(array,b,a):


        N = len(array)
        indexes = range(0,N)

        for i in range(0,N):
            if array[i] == 0:
                indexes[i] = -1


        splitdata = [x[x!=0] for x in np.split(array, np.where(array==0)[0]) if len(x[x!=0])]
        splitIndexes = [x[x!=-1] for x in np.split(indexes, np.where(indexes==-1)[0]) if len(x[x!=-1])]


        filtValues_section=list()
        for data in splitdata:
            padlen = 3 * max(len(a), len(b)) # default as defined in https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
            if len(data) <= padlen:
                padlen = len(data) - 1
            filtValues_section.append(signal.filtfilt(b, a, data ,padlen=padlen,axis=0))

        indexes = np.concatenate(splitIndexes)
        values = np.concatenate(filtValues_section)

        out = np.zeros((N))
        j = 0
        for i in indexes:
            out[i] = values[j]
            j+=1
        return out


    fp=btkAcq.GetPointFrequency()
    bPoint, aPoint = signal.butter(order, fc / (fp*0.5) , btype='lowpass')

    for pointIt in btk.Iterate(btkAcq.GetPoints()):
        if pointIt.GetType() == btk.btkPoint.Marker and pointIt.GetLabel() in markers:
            label = pointIt.GetLabel()
            if zerosFiltering:
                x = filterZeros(pointIt.GetValues()[:,0],bPoint,aPoint)
                y = filterZeros(pointIt.GetValues()[:,1],bPoint,aPoint)
                z = filterZeros(pointIt.GetValues()[:,2],bPoint,aPoint)
            else:
                x=signal.filtfilt(bPoint, aPoint, pointIt.GetValues()[:,0],axis=0  )
                y=signal.filtfilt(bPoint, aPoint, pointIt.GetValues()[:,1],axis=0  )
                z=signal.filtfilt(bPoint, aPoint, pointIt.GetValues()[:,2],axis=0  )

            btkAcq.GetPoint(label).SetValues(np.array( [x,y,z] ).transpose())


            # pointIt.SetValues(np.array( [x,y,z] ).transpose())


def forcePlateFiltering(btkAcq,order=4, fc =5):
    """
        Low-pass filtering of all points in an acquisition

        :Parameters:
            - `btkAcq` (btkAcquisition) - btk acquisition instance
            - `fc` (double) - cut-off frequency
            - `order` (double) - order of the low-pass filter
   """


    fp=btkAcq.GetAnalogFrequency()
    bPoint, aPoint = signal.butter(order, fc / (fp*0.5) , btype='lowpass')

    # --- ground reaction force wrench ---
    pfe = btk.btkForcePlatformsExtractor()
    pfe.SetInput(btkAcq)
    pfc = pfe.GetOutput()
    pfc.Update()

    for i in range(0,pfc.GetItemNumber()):



        for j in range(0,pfc.GetItem(i).GetChannelNumber()):

            values = pfc.GetItem(i).GetChannel(j).GetValues()[:,0]

            values_filt = signal.filtfilt(bPoint, aPoint, values,axis=0  )

            label = pfc.GetItem(i).GetChannel(j).GetLabel() # SetValues on channel not store new values
            try:
                btkAcq.GetAnalog(label).SetValues(values_filt)
            except RuntimeError:
                logging.error("[pyCGM2] filtering of the force place %i impossible - label %s not found"%(i,label))

# ----- methods ---------
def arrayLowPassFiltering(valuesArray, freq, order=2, fc =6):
    """
        low-pass filtering of an numpy array

        :Parameters:
             - `valuesArray` (numpy.array(n,n)) - array
            - `fc` (double) - cut-off frequency
            - `order` (double) - order of the low-pass filter
    """
    b, a = signal.butter(order, fc / (freq*0.5) , btype='lowpass')

    out = np.zeros(valuesArray.shape)
    for i in range(0, valuesArray.shape[1]):
        out[:,i] = signal.filtfilt(b, a, valuesArray[:,i] )

    return out

def psd(x, fs=1.0, window='hanning', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', show=True, ax=None, scales='linear', xlim=None,
        units='V'):

    def _plot(x, fs, f, P, mpf, fmax, fpcntile, scales, xlim, units, ax):
        """Plot results of the ellipse function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            if scales.lower() == 'semilogy' or scales.lower() == 'loglog':
                ax.set_yscale('log')
            if scales.lower() == 'semilogx' or scales.lower() == 'loglog':
                ax.set_xscale('log')
            plt.plot(f, P, linewidth=2)
            ylim = ax.get_ylim()
            plt.plot([fmax, fmax], [np.max(P), np.max(P)], 'ro',
                     label='Fpeak  = %.2f' % fmax)
            plt.plot([fpcntile[50], fpcntile[50]], ylim, 'r', lw=1.5,
                     label='F50%%   = %.2f' % fpcntile[50])
            plt.plot([mpf, mpf], ylim, 'r--', lw=1.5,
                     label='Fmean = %.2f' % mpf)
            plt.plot([fpcntile[95], fpcntile[95]], ylim, 'r-.', lw=2,
                     label='F95%%   = %.2f' % fpcntile[95])
            leg = ax.legend(loc='best', numpoints=1, framealpha=.5,
                            title='Frequencies [Hz]')
            plt.setp(leg.get_title(), fontsize=12)
            plt.xlabel('Frequency [$Hz$]', fontsize=12)
            plt.ylabel('Magnitude [%s$^2/Hz$]' % units, fontsize=12)
            plt.title('Power spectral density', fontsize=12)
            if xlim:
                ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.tight_layout()
            plt.grid()
            plt.show()

    """Estimate power spectral density characteristcs using Welch's method.

    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = 'tnorm.py v.1 2013/09/16'

    This function is just a wrap of the scipy.signal.welch function with
    estimation of some frequency characteristcs and a plot. For completeness,
    most of the help from scipy.signal.welch function is pasted here.

    Welch's method [1]_ computes an estimate of the power spectral density
    by dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the periodograms.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series in units of Hz. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length will be used for nperseg.
        Defaults to 'hanning'.
    nperseg : int, optional
        Length of each segment.  Defaults to half of `x` length.
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg / 2``.  Defaults to None.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired.  If None,
        the FFT length is `nperseg`. Defaults to None.
    detrend : str or function, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`. If it is a
        function, it takes a segment and returns a detrended segment.
        Defaults to 'constant'.
    show : bool, optional (default = False)
        True (1) plots data in a matplotlib figure.
        False (0) to not plot.
    ax : a matplotlib.axes.Axes instance (default = None)
    scales : str, optional
        Specifies the type of scale for the plot; default is 'linear' which
        makes a plot with linear scaling on both the x and y axis.
        Use 'semilogy' to plot with log scaling only on the y axis, 'semilogx'
        to plot with log scaling only on the x axis, and 'loglog' to plot with
        log scaling on both the x and y axis.
    xlim : float, optional
        Specifies the limit for the `x` axis; use as [xmin, xmax].
        The defaukt is `None` which sets xlim to [0, Fniquist].
    units : str, optional
        Specifies the units of `x`; default is 'V'.

    Returns
    -------
    Fpcntile : 1D array
        frequency percentiles of the power spectral density
        For example, Fpcntile[50] gives the median power frequency in Hz.
    mpf : float
        Mean power frequency in Hz.
    fmax : float
        Maximum power frequency in Hz.
    Ptotal : float
        Total power in `units` squared.
    f : 1D array
        Array of sample frequencies in Hz.
    P : 1D array
        Power spectral density or power spectrum of x.

    See Also
    --------
    scipy.signal.welch

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements.  For the default 'hanning' window an
    overlap of 50% is a reasonable trade off between accurately estimating
    the signal power, while not over counting any of the data.  Narrower
    windows may require a larger overlap.
    If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.

    Examples (also from scipy.signal.welch)
    --------
    >>> import numpy as np
    >>> from psd import psd
    #Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
    # 0.001 V**2/Hz of white noise sampled at 10 kHz and calculate the PSD:
    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2*np.sqrt(2)
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> x = amp*np.sin(2*np.pi*freq*time)
    >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> psd(x, fs=freq);
    """


    if not nperseg:
        nperseg = np.ceil(len(x) / 2)
    f, P = signal.welch(x, fs, window, nperseg, noverlap, nfft, detrend)
    Area = integrate.cumtrapz(P, f, initial=0)
    Ptotal = Area[-1]
    mpf = integrate.trapz(f * P, f) / Ptotal  # mean power frequency
    fmax = f[np.argmax(P)]
    # frequency percentiles
    inds = [0]
    Area = 100 * Area / Ptotal  # + 10 * np.finfo(np.float).eps
    for i in range(1, 101):
        inds.append(np.argmax(Area[inds[-1]:] >= i) + inds[-1])
    fpcntile = f[inds]

    if show:
        _plot(x, fs, f, P, mpf, fmax, fpcntile, scales, xlim, units, ax)

    return fpcntile, mpf, fmax, Ptotal, f, P
