

import numpy as np

import pyCGM2
LOGGER = pyCGM2.LOGGER
import matplotlib.pyplot as plt

from typing import Optional


def detectStart_fromThreshold(values,reference, type = "lower",epsilon=0.05,firstFrame=0,nppf=1):

    if type == "lower":
        inds = np.where(values < (1-epsilon)*reference)
    elif type == "greater":
        inds = np.where(values > (1+epsilon)*reference)
    else:
        raise Exception( "type not known, greater or lower")

    i0 = inds[0][0]/nppf+firstFrame


    return int(i0)


# ----------------detecta package------------------------------------------
def detect_cusum(x:np.ndarray, threshold:int=1, drift:int=0, ending:bool=False, show:bool=True, ax:Optional[plt.Axes]=None):
    """
    Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

    Args:
        x (np.ndarray): Data in which to detect changes.
        threshold (int, optional): Amplitude threshold for the change in the data. Default is 1.
        drift (int, optional): Drift term that prevents any change in the absence of change. Default is 0.
        ending (bool, optional): If True, estimates when the change ends. If False, does not. Default is False.
        show (bool, optional): If True, plots data in matplotlib figure. If False, does not. Default is True.
        ax (plt.Axes, optional): A matplotlib Axes instance for the plot. Default is None.

    Returns:
        tuple: A tuple containing:
            - ta (np.ndarray): Alarm time (index of when the change was detected).
            - tai (np.ndarray): Index of when the change started.
            - taf (np.ndarray): Index of when the change ended (if `ending` is True).
            - amp (np.ndarray): Amplitude of changes (if `ending` is True).

    Notes:
        - Tuning of the CUSUM algorithm according to Gustafsson (2000): Start with a very large `threshold`. Choose `drift` to half the expected change, then adjust so that g = 0 more than 50% of the time. Set the `threshold` for the desired false alarm rate or detection delay. Decrease `drift` for faster detection, increase for fewer false alarms.
        - By default, repeated sequential changes are not deleted. Set `ending` to True to delete them.
        - See the referenced IPython Notebook for more information.

    References:
        - Gustafsson (2000) Adaptive Filtering and Change Detection.
        - [CUSUM Notebook](https://github.com/demotu/detecta/blob/master/docs/detect_cusum.ipynb)

    Examples:
        >>> x = np.random.randn(300)/5
        >>> x[100:200] += np.arange(0, 4, 4/100)
        >>> ta, tai, taf, amp = detect_cusum(x, 2, .02, True, True)
        >>> x = np.random.randn(300)
        >>> x[100:200] += 6
        >>> detect_cusum(x, 4, 1.5, True, True)
        >>> x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
        >>> ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)

    Version history:
        - '1.0.5': Part of the detecta module - [Detecta PyPI](https://pypi.org/project/detecta/)
    """

    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _ = detect_cusum(x[::-1], threshold, drift, show=False)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = np.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = np.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[np.argmax(taf >= i) for i in ta]]
            else:
                ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~np.append(False, ind)]
            tai = tai[~np.append(False, ind)]
            taf = taf[~np.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]

    if show:
        _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn)

    return ta, tai, taf, amp


    def _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn):
        """Plot results of the detect_cusum function, see its help."""

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
            return

        if ax is None:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        else:
            if len(ax) != 2:
                print("Plotting failed. Expecting instances of 2 matplotlib axes.")
                return
            (ax1, ax2) = ax

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                        label='Start')
            if ending:
                ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                            label='Ending')
            ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                        label='Alarm')
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        ax1.set_xlim(-.01*x.size, x.size*1.01-1)
        ax1.set_xlabel('Data #', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax1.set_title('Time series and detected changes ' +
                        '(threshold= %.3g, drift= %.3g): N changes = %d'
                        % (threshold, drift, len(tai)))
        ax2.plot(t, gp, 'y-', label='+')
        ax2.plot(t, gn, 'm-', label='-')
        ax2.set_xlim(-.01*x.size, x.size*1.01-1)
        ax2.set_xlabel('Data #', fontsize=14)
        ax2.set_ylim(-0.01*threshold, 1.1*threshold)
        ax2.axhline(threshold, color='r')
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax2.set_title('Time series of the cumulative sums of ' +
                        'positive and negative changes')
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        plt.tight_layout()
        plt.show()


def detect_onset(x, threshold=0, n_above=1, n_below=0,
                 threshold2=None, n_above2=1, show=False, ax=None):
    """
    Detects onset in data based on amplitude threshold.

    Args:
        x (np.ndarray): Data to analyze.
        threshold (float, optional): Minimum amplitude of `x` to detect. Default is 0.
        n_above (int, optional): Minimum number of continuous samples >= `threshold` to detect. Default is 1.
        n_below (int, optional): Minimum number of continuous samples below `threshold` that will be ignored in the detection. Default is 0.
        threshold2 (float, optional): Minimum amplitude of `n_above2` values in `x` to detect. Default is None.
        n_above2 (int, optional): Minimum number of samples >= `threshold2` to detect. Default is 1.
        show (bool, optional): If True, plots data in matplotlib figure. If False, doesn't plot. Default is False.
        ax (plt.Axes, optional): A matplotlib Axes instance for the plot. Default is None.

    Returns:
        np.ndarray: 2D array [indi, indf] with initial and final indices of the onset events.

    Notes:
        - Signal-to-noise characteristic of the data might require tuning of parameters.
        - See the referenced IPython Notebook for more information.

    References:
        - [Detect Onset Notebook](https://github.com/demotu/detecta/blob/master/docs/detect_onset.ipynb)

    Examples:
        >>> x = np.random.randn(200)/10
        >>> # various examples using the function with different parameters

    Version history:
        - '1.0.7': Part of the detecta module - [Detecta PyPI](https://pypi.org/project/detecta/)
        - '1.0.6': Deleted 'from __future__ import', added parameters `threshold2` and `n_above2`
    """
    def _plot(x, threshold, n_above, n_below, threshold2, n_above2, inds, ax):
        """Plot results of the detect_onset function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 4))

            if inds.size:
                for (indi, indf) in inds:
                    if indi == indf:
                        ax.plot(indf, x[indf], 'ro', mec='r', ms=6)
                    else:
                        ax.plot(range(indi, indf+1), x[indi:indf+1], 'r', lw=1)
                        ax.axvline(x=indi, color='b', lw=1, ls='--')
                    ax.axvline(x=indf, color='b', lw=1, ls='--')
                inds = np.vstack((np.hstack((0, inds[:, 1])),
                                np.hstack((inds[:, 0], x.size-1)))).T
                for (indi, indf) in inds:
                    ax.plot(range(indi, indf+1), x[indi:indf+1], 'k', lw=1)
            else:
                ax.plot(x, 'k', lw=1)
                ax.axhline(y=threshold, color='r', lw=1, ls='-')

            ax.set_xlim(-.02*x.size, x.size*1.02-1)
            ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            yrange = ymax - ymin if ymax > ymin else 1
            ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
            ax.set_xlabel('Data #', fontsize=14)
            ax.set_ylabel('Amplitude', fontsize=14)
            if threshold2 is not None:
                text = 'threshold=%.3g, n_above=%d, n_below=%d, threshold2=%.3g, n_above2=%d'
            else:
                text = 'threshold=%.3g, n_above=%d, n_below=%d, threshold2=%r, n_above2=%d'            
            ax.set_title(text % (threshold, n_above, n_below, threshold2, n_above2))
            # plt.grid()
            plt.show()

    x = np.atleast_1d(x).astype('float64')
    # deal with NaN's (by definition, NaN's are not greater than threshold)
    x[np.isnan(x)] = -np.inf
    # indices of data greater than or equal to threshold
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) > n_below+1], \
                          inds[np.diff(np.hstack((inds, np.inf))) > n_below+1])).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1]-inds[:, 0] >= n_above-1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if np.count_nonzero(x[inds[i, 0]: inds[i, 1]+1] >= threshold2) < n_above2:
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])  # standardize inds shape for output
    if show and x.size > 1:  # don't waste my time ploting one datum
        _plot(x, threshold, n_above, n_below, threshold2, n_above2, inds, ax)

    return inds


    


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, title=True):

    """
    Detects peaks in data based on their amplitude and other features.

    Args:
        x (np.ndarray): Data to analyze.
        mph (float, optional): Detect peaks that are greater than minimum peak height or valleys smaller than maximum peak height if `valley` is True. Default is None.
        mpd (int, optional): Minimum number of data points separating peaks. Default is 1.
        threshold (float, optional): Detect peaks that are greater than the threshold in relation to their neighbors. Default is 0.
        edge (str, optional): For a flat peak, keep only the specified edges. Options are 'rising', 'falling', 'both', or None. Default is 'rising'.
        kpsh (bool, optional): Keep peaks with the same height even if they are closer than `mpd`. Default is False.
        valley (bool, optional): If True, detect valleys instead of peaks. Default is False.
        show (bool, optional): If True, plot data using matplotlib. Default is False.
        ax (plt.Axes, optional): Matplotlib Axes instance for plotting. Default is None.
        title (bool or str, optional): Show standard title, custom title, or no title in the plot. Default is True.

    Returns:
        np.ndarray: Indices of the detected peaks in `x`.

    Notes:
        - The detection of valleys is done by negating the data: `ind_valleys = detect_peaks(-x)`.
        - The function can handle NaN values.
        - See the IPython Notebook for more information.

    References:
        - [Detect Peaks Notebook](https://github.com/demotu/detecta/blob/master/docs/detect_peaks.ipynb)

    Examples:
        >>> x = np.random.randn(100)
        >>> # various examples using the function with different parameters

    Version history:
        - '1.0.7': Part of the detecta module - [Detecta PyPI](https://pypi.org/project/detecta/)
        - '1.0.6': Fixes and new features regarding ax object and title parameter.
        - '1.0.5': Inversion of `mph` sign based on `valley` parameter.
    """
    def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title):
        """Plot results of the detect_peaks function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 4))
                no_ax = True
            else:
                no_ax = False

            ax.plot(x, 'b', lw=1)
            if ind.size:
                label = 'valley' if valley else 'peak'
                label = label + 's' if ind.size > 1 else label
                ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                        label='%d %s' % (ind.size, label))
                ax.legend(loc='best', framealpha=.5, numpoints=1)
            ax.set_xlim(-.02*x.size, x.size*1.02-1)
            ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            yrange = ymax - ymin if ymax > ymin else 1
            ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
            ax.set_xlabel('Data #', fontsize=14)
            ax.set_ylabel('Amplitude', fontsize=14)
            if title:
                if not isinstance(title, str):
                    mode = 'Valley detection' if valley else 'Peak detection'
                    title = "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"% \
                            (mode, str(mph), mpd, str(threshold), edge)
                ax.set_title(title)
            # plt.grid()
            if no_ax:
                plt.show()

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind


    

def detect_seq(x, value=np.nan, index=False, min_seq=1, max_alert=0,
               show=False, ax=None):
    """
    Detects indices in x of sequential data identical to a specified value.

    Args:
        x (np.ndarray): Data to analyze.
        value (float, optional): Value to be found in data. Default is np.nan.
        index (bool, optional): If True, returns 2D array of initial and final indices where data equals value. If False, returns 1D array of Boolean values. Default is False.
        min_seq (int, optional): Minimum number of sequential values to detect. Default is 1.
        max_alert (int, optional): Minimal number of sequential data for an alert message. Set to 0 to disable alerts. Default is 0.
        show (bool, optional): If True, plots the data. Default is False.
        ax (Optional[plt.Axes], optional): Matplotlib axis object for plotting. Default is None.

    Returns:
        np.ndarray: Either a 2D array [indi, indf] of initial and final indices (if index=True) or a 1D array of Boolean values (if index=False).

    References:
        - [Detect Seq Notebook](https://github.com/demotu/detecta/blob/master/docs/detect_seq.ipynb)

    Examples:
        >>> x = [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]
        >>> # various examples using the function with different parameters

    Version history:
        - '1.0.1': Part of the detecta module - [Detecta PyPI](https://pypi.org/project/detecta/)
    """
    def _plot(x, value, min_seq, ax, idx):
        """Plot results of the detect_seq function, see its help.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            x = np.asarray(x)
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 4))

            if idx.size:
                for (indi, indf) in idx:
                    if indi == indf:
                        ax.plot(indf, x[indf], 'ro', mec='r', ms=6)
                    else:
                        ax.plot(range(indi, indf+1), x[indi:indf+1], 'b', lw=1)
                        ax.axvline(x=indi, color='r', lw=1, ls='--')
                        ax.plot(indi, x[indi], 'r>', mec='r', ms=6)
                        ax.plot(indf, x[indf], 'r<', mec='r', ms=6)
                    ax.axvline(x=indf, color='r', lw=1, ls='--')
                idx = np.vstack((np.hstack((0, idx[:, 1])),
                                np.hstack((idx[:, 0], x.size-1)))).T
                for (indi, indf) in idx:
                    ax.plot(range(indi, indf+1), x[indi:indf+1], 'k', lw=1)
            else:
                ax.plot(x, 'k', lw=1)

            ax.set_xlim(-.02*x.size, x.size*1.02-1)
            ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            yrange = ymax - ymin if ymax > ymin else 1
            ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
            text = 'Value=%.3g, minimum number=%d'
            ax.set_title(text % (value, min_seq))
            plt.show()

    idx = np.r_[False, np.isnan(x) if np.isnan(value) else np.equal(x, value), False]

    if index or min_seq > 1 or max_alert or show:
        idx2 = np.where(np.abs(np.diff(idx))==1)[0].reshape(-1, 2)
        if min_seq > 1:
            idx2 = idx2[np.where(np.diff(idx2, axis=1) >= min_seq)[0]]
            if not index:
                idx = idx[1:-1]*False
                for i, f in idx2:
                    idx[i:f] = True           
        idx2[:, 1] = idx2[:, 1] - 1

    if index:
        idx = idx2
    elif len(idx) > len(x):
        idx = idx[1:-1]

    if max_alert and idx2.shape[0]:
        seq = np.diff(idx2, axis=1)
        for j in range(idx2.shape[0]):
            bitlen = seq[j]
            if bitlen >= max_alert:
                text = 'Sequential data equal or longer than {}: ({}, {})'
                print(text.format(max_alert, bitlen, idx2[j]))

    if show:
        _plot(x, value, min_seq, ax, idx2)
        
    return idx


    
