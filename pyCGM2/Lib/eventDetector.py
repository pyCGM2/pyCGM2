# -*- coding: utf-8 -*-
from pyCGM2.Events import events
from pyCGM2.Signal import signal_processing

def zeni(acqGait,footStrikeOffset=0,footOffOffset=0,**kwargs):
    """
    Detect gait event according Zeni's algorithm (Coordinate only method)

    :param acqGait [Btk.Acquisition]: gait acquisition

    **optional**
    :param footStrikeOffset [int]:  systematic offset to add to foot strike
    :param footOffOffset [int]: systematic oofset to add to foot off

    **Return**
    :param AcqGait [Btk.Acquisition]:  gait acquisition updated with events


    :example:

    >>>
    """
    acqGait.ClearEvents()


    if "fc_lowPass_marker" in kwargs.keys() and kwargs["fc_lowPass_marker"]!=0 :
        fc = kwargs["fc_lowPass_marker"]
        order = 4
        if "order_lowPass_marker" in kwargs.keys():
            order = kwargs["order_lowPass_marker"]
        signal_processing.markerFiltering(acqGait,["LPSI","RPSI","LHEE","LTOE","RHEE","RTOE"],order=order, fc =fc)


    # ----------------------EVENT DETECTOR-------------------------------
    evp = events.ZeniProcedure()
    evp.setFootStrikeOffset(footStrikeOffset)
    evp.setFootOffOffset(footOffOffset)

    # event filter
    evf = events.EventFilter(evp,acqGait)
    evf.detect()
    state = evf.getState()
    return acqGait,state
