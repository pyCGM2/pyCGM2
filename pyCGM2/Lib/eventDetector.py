# -*- coding: utf-8 -*-
from pyCGM2.Events import events

def zeni(acqGait,footStrikeOffset=0,footOffOffset=0):
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
    # ----------------------EVENT DETECTOR-------------------------------
    evp = events.ZeniProcedure()
    evp.setFootStrikeOffset(footStrikeOffset)
    evp.setFootOffOffset(footOffOffset)

    # event filter
    evf = events.EventFilter(evp,acqGait)
    evf.detect()
    state = evf.getState()
    return acqGait,state


def deepevent(acqGait):
    acqGait.ClearEvents()
    # ----------------------EVENT DETECTOR-------------------------------
    evp = events.DeepEventProcedure()

    # event filter
    evf = events.EventFilter(evp,acqGait)
    evf.detect()
    state = evf.getState()
    return acqGait,state
