# coding: utf-8
# pytest -s --disable-pytest-warnings  test_openmaTools.py::Test_openma::test_reader
from __future__ import unicode_literals
import pyCGM2
from pyCGM2.Utils import testingUtils,files
import ipdb
import os
import logging
from pyCGM2.Eclipse import vskTools,eclipse
from pyCGM2 import enums

import pytest
from pyCGM2.Tools import trialTools
from pyCGM2.Utils import utils
from pyCGM2 import btk


class Test_openma:
    def test_reader(self):
        trial = trialTools.smartTrialReader(pyCGM2.TEST_DATA_PATH +"LowLevel\\IO\\Hånnibøl_c3d\\","static.c3d")
