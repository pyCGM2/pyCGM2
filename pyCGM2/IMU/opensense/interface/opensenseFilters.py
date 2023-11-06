# -*- coding: utf-8 -*-
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Tools import  btkTools,opensimTools
import os
import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files

import btk
import opensim


class opensenseInterfaceImuPlacerFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure


    def run(self):
        self.m_procedure.run()


class opensenseInterfaceImuInverseKinematicFilter(object):
    def __init__(self,procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()