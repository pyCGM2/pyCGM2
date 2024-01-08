"""
The module contains filter for detecting foot contact events.

check out the script : *\Tests\test_events.py* for examples
"""
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Events.eventProcedures import EventProcedure
import btk

from typing import List, Tuple, Dict, Optional,Union

class EventFilter(object):
    """
    Event filter to handle an event procedure.

    This filter is designed to detect foot contact events within a motion capture acquisition. It uses a specified event procedure to determine the timings of these events.

    Args:
        procedure (EventProcedure): An event procedure instance used for detecting foot contact events.
        acq (btk.btkAcquisition): A BTK acquisition instance containing motion capture data.
    """
    def __init__(self,procedure:EventProcedure,acq:btk.btkAcquisition):
        """Initializes the EventFilter with a given event procedure and acquisition data"""

        self.m_aqui = acq
        self.m_procedure = procedure
        self.m_state = None

    def getState(self):
        """
        Return the state of the filter.

        Indicates whether the event detection procedure was successful or not.

        Returns:
            bool: The state of the filter, True if event detection was successful, otherwise False.
        """
        return self.m_state

    def detect(self):
        """
        Run the event detection procedure.

        This method applies the event detection procedure to the acquisition data to identify foot strike and foot off events. Detected events are added to the acquisition instance.
        """
        pf = self.m_aqui.GetPointFrequency()

        eventDescriptor = self.m_procedure.description

        if self.m_procedure.detect(self.m_aqui) == 0:
            self.m_state = False
        else:
            indexes_fs_left, indexes_fo_left, indexes_fs_right, indexes_fo_right =  self.m_procedure.detect(self.m_aqui)
            self.m_state = True
            for ind in indexes_fs_left:
                ev = btk.btkEvent('Foot Strike', (ind-1)/pf, 'Left', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetFrame(int(ind))
                ev.SetId(1)
                self.m_aqui.AppendEvent(ev)

            for ind in indexes_fo_left:
                ev = btk.btkEvent('Foot Off', (ind-1)/pf, 'Left', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetFrame(int(ind))
                ev.SetId(2)
                self.m_aqui.AppendEvent(ev)

            for ind in indexes_fs_right:
                ev = btk.btkEvent('Foot Strike', (ind-1)/pf, 'Right', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetFrame(int(ind))
                ev.SetId(1)
                self.m_aqui.AppendEvent(ev)

            for ind in indexes_fo_right:
                ev = btk.btkEvent('Foot Off', (ind-1)/pf, 'Right', btk.btkEvent.Manual, '', eventDescriptor)
                ev.SetFrame(int(ind))
                ev.SetId(2)
                self.m_aqui.AppendEvent(ev)
