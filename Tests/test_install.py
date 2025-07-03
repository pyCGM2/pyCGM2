# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_anomalies.py::Test_markerAnomalies::test_anomalies

import pytest


class Test_install:
    def test_import(self):
        import pyCGM2
        import btk
        import opensim

class Test_CGM:
    def test_cgm23(self):
        pass


class Test_dataProcess:
    def test_dataExport(self):
        pass

