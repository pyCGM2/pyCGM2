# -*- coding: utf-8 -*-
"""Logging utilities of pyCGM2."""
import logging

_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"

class pyCGM2_Logger(object):
    """Wrapper around a :class:`logging.Logger` with the pyCGM2 message format.

    The logger outputs to the console at INFO level by default.

    Args:
        name (str): Name of the logger.
    """

    def __init__(self,name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.__get_stream_handler())

    def getLogger(self):
        """Return the wrapped logger.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    def __get_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(_log_format))
        return stream_handler

    def set_file_handler(self,filename):
        """Add a file handler writing log messages into a file (overwritten).

        Args:
            filename (str): Path of the log file.
        """
        file_handler = logging.FileHandler(filename,mode='w')
        file_handler.setFormatter(logging.Formatter(_log_format))
        self.logger.addHandler(file_handler)

    def setLevel(self,level):
        """Set the logging level.

        Args:
            level (str): One of ``"error"``, ``"warning"``, ``"info"`` or ``"debug"``.
        """
        if level == "error":
            self.logger.setLevel(logging.ERROR)
        if level == "warning":
            self.logger.setLevel(logging.WARNING)
        if level == "info":
            self.logger.setLevel(logging.INFO)
        if level == "debug":
            self.logger.setLevel(logging.DEBUG)
