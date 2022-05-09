# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Functions
#APIDOC["Draft"]=False
#--end--
from pyCGM2.Events import eventFilters
from pyCGM2.Events import eventProcedures
from pyCGM2.Signal import signal_processing


def zeni(acqGait, footStrikeOffset=0, footOffOffset=0, **kwargs):
    """kinematic-based event detector according Zeni et al(2008).

    This method need the presence of the markers "LPSI","RPSI","LHEE","LTOE","RHEE","RTOE"


    *Reference:*
    Zeni, J. A.; Richards, J. G.; Higginson, J. S. (2008) Two simple methods for determining gait events during treadmill and overground walking using kinematic data. In : Gait & posture, vol. 27, n° 4, p. 710–714. DOI: 10.1016/j.gaitpost.2007.07.007.

    Args:
        acqGait (btk.Acquisition): an acquisition instance.
        footStrikeOffset (int): systematic offset to add to all `footStrikeOffset` events. Default is 0.
        footOffOffset (int): systematic offset to add to all `footOffOffset` events. Default is 0.

    Keyword Arguments:
        fc_lowPass_marker (double) : cut-off frequency of the lowpass filter applied on markers
        order_lowPass_marker (int): order of the lowpass filter applied on markers

    Returns:
        acqGait ( btk.Acquisition): updated acquisition with detected events.
        state (bool): state of the detector


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
