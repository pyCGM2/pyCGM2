# -*- coding: utf-8 -*-
"""
Obsolete module : work with anomaly and inspector modules instead
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER


class QualityFilter(object):
    """Filter running a quality-check procedure of the :mod:`pyCGM2.Inspect.inspectProcedures` module.

    Args:
        procedure: Quality-check procedure instance.
        verbose (bool, optional): Log the result of the test. Defaults to True.
    """
    def __init__(self,procedure,verbose=True):
        self.m_procedure = procedure
        self.exceptionMode = False
        self.verbose = verbose

    def enableExceptionMode(self):
        """Enable the exception mode: a failed test raises an exception."""
        self.m_procedure.exceptionMode = True
        self.exceptionMode = True

    def disableExceptionMode(self):
        """Disable the exception mode: a failed test is only logged."""
        self.m_procedure.exceptionMode = False
        self.exceptionMode = False

    def getState(self):
        """Return the state of the procedure.

        Returns:
            bool: True if the test succeeded.
        """
        return self.m_procedure.state

    def run(self):
        """Run the quality check.

        Raises:
            Exception: If the test fails and the exception mode is enabled.
        """

        self.m_procedure.check()

        if self.verbose:
            LOGGER.logger.info("----Quality test: %s ----"%(self.m_procedure.title))
            if self.m_procedure.state:
                LOGGER.logger.info("quality Test  => OK :-)")
            else :
                if not self.exceptionMode:
                    LOGGER.logger.info("quality Test => FAILED :-(")
                else:
                    raise Exception ("[pyCGM2] Quality test (%s) fails"%(self.m_procedure.title))
            LOGGER.logger.info("--------------------------")
