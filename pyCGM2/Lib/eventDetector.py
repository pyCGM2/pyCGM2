from pyCGM2.Events import eventFilters
from pyCGM2.Events import eventProcedures
from pyCGM2.Signal import signal_processing
import btk

from typing import List, Tuple, Dict, Optional,Union

def zeni(acqGait:btk.btkAcquisition, 
         footStrikeOffset:int=0, 
         footOffOffset:int=0,
         **kwargs):
    """
    Kinematic-based gait event detection according to Zeni et al. (2008).

    This function detects gait events in a BTK acquisition instance using marker data. 
    It requires the presence of specific markers and can apply a low-pass filter to marker data. 
    The method is based on the approach described by Zeni, J. A., Richards, J. G., 
    & Higginson, J. S. in their 2008 paper.

    Args:
        acqGait (btk.btkAcquisition): An acquisition instance with gait data.
        footStrikeOffset (int, optional): A systematic offset added to all foot strike events. Defaults to 0.
        footOffOffset (int, optional): A systematic offset added to all foot off events. Defaults to 0.

    Keyword Args:
        fc_lowPass_marker (float): Cut-off frequency of the low-pass filter applied to markers. If not specified or 0, no filtering is applied.
        order_lowPass_marker (int): Order of the low-pass filter applied to markers. Defaults to 4 if not specified.

    Returns:
        Tuple[btk.btkAcquisition, bool]: A tuple containing the updated acquisition instance with detected events, and a boolean indicating the state of the detector.

    Example:
        >>> updated_acq, detection_state = zeni(acquisition, footStrikeOffset=10, footOffOffset=5)

    Reference:
        Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008). Two simple methods for determining gait events 
        during treadmill and overground walking using kinematic data. Gait & Posture, 27(4), 710–714. 
        DOI: 10.1016/j.gaitpost.2007.07.007.
    """

    acqGait.ClearEvents()

    if "fc_lowPass_marker" in kwargs.keys() and kwargs["fc_lowPass_marker"] != 0:
        fc = kwargs["fc_lowPass_marker"]
        order = 4
        if "order_lowPass_marker" in kwargs.keys():
            order = kwargs["order_lowPass_marker"]
        signal_processing.markerFiltering(
            acqGait, ["LPSI", "RPSI", "LHEE", "LTOE", "RHEE", "RTOE"], order=order, fc=fc)

    # ----------------------EVENT DETECTOR-------------------------------
    evp = eventProcedures.ZeniProcedure()
    evp.setFootStrikeOffset(footStrikeOffset)
    evp.setFootOffOffset(footOffOffset)

    # event filter
    evf = eventFilters.EventFilter(evp, acqGait)
    evf.detect()
    state = evf.getState()
    return acqGait, state
