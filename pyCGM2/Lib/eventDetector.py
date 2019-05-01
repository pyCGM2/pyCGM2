# -*- coding: utf-8 -*-
from pyCGM2.Events import events

def zeni(acqGait,footStrikeOffset=0,footOffOffset=0):

    acqGait.ClearEvents()
    # ----------------------EVENT DETECTOR-------------------------------
    evp = events.ZeniProcedure()
    evp.setFootStrikeOffset(footStrikeOffset)
    evp.setFootOffOffset(footOffOffset)

    # event filter
    evf = events.EventFilter(evp,acqGait)
    evf.detect()

    return acqGait
