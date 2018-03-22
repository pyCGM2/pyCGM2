import logging
import sys
from logging.handlers import RotatingFileHandler
from logging import handlers
#logging.basicConfig(filename = "installer.log", level=logging.DEBUG)

def setLoggingLevel(level):
    logging.basicConfig(format = "[pyCGM2-%(levelname)s]-%(module)s-%(funcName)s : %(message)s",level = level)

def setLogger(filename = "pyCGM2.log",level = logging.INFO ):
    log = logging.getLogger('')
    log.setLevel(level)
    format = logging.Formatter("[pyCGM2-%(levelname)s]-%(module)s-%(funcName)s : %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)
    fh = handlers.RotatingFileHandler(filename, maxBytes=(1048576*5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)
