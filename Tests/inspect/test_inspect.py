# -*- coding: utf-8 -*-
import ipdb
import logging
import os

# pyCGM2 settings
import pyCGM2
from pyCGM2.Inspect import inspectFilters, inspectProcedures
from pyCGM2.Tools import btkTools


from pyCGM2 import log;
log.setLogger(filename = "inspector.log",level = logging.INFO)
with open('inspector.log', 'w'):   pass



class Tests():

    @classmethod
    def test(cls):
        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\inspection\\sample0\\"
        filename = "static.c3d"


        logging.info("TEST")



        mp={
        'Bodymass'   : 70.0,
        'LeftLegLength' : 900.0,
        'RightLegLength' : 900.0 ,
        'LeftKneeWidth' : 100.0,
        'RightKneeWidth' : 100.0,
        'LeftAnkleWidth' : 70.0,
        'RightAnkleWidth' : 70.0,
        'LeftSoleDelta' : 0,
        'RightSoleDelta' : 0,
        'LeftShoulderOffset'   : 0,
        'LeftElbowWidth' : 0,
        'LeftWristWidth' : 0 ,
        'LeftHandThickness' : 0 ,
        'RightShoulderOffset'   : 0,
        'RightElbowWidth' : 0,
        'RightWristWidth' : 0 ,
        'RightHandThickness' : 0}

        inspectprocedure1 = inspectProcedures.AnthropometricDataQualityProcedure(mp)



        inspector = inspectFilters.QualityFilter(inspectprocedure1)
        inspector.run()


        acq = btkTools.smartReader(DATA_PATH+"gait_gap.c3d")
        inspectprocedure1 = inspectProcedures.GapQualityFilter(acq)
        inspector = inspectFilters.QualityFilter(inspectprocedure1)
        inspector.run()


        acq = btkTools.smartReader(DATA_PATH+"gait_swapped.c3d")
        inspectprocedure2 = inspectProcedures.SwappingMarkerQualityProcedure(acq)
        inspector = inspectFilters.QualityFilter(inspectprocedure2)
        inspector.run()

        # acq = btkTools.smartReader(DATA_PATH+"gait_noEvents.c3d")
        # inspectprocedure2 = inspectProcedures.GaitEventQualityProcedure(acq)
        # inspector = inspectFilters.QualityFilter(inspectprocedure2)
        # inspector.run()


        acq = btkTools.smartReader(DATA_PATH+"gait_wrongEvent.c3d")
        inspectprocedure2 = inspectProcedures.GaitEventQualityProcedure(acq)
        inspector = inspectFilters.QualityFilter(inspectprocedure2)
        inspector.run()


        acq = btkTools.smartReader(DATA_PATH+"gait_PelvisSwapped.c3d")
        inspectprocedure2 = inspectProcedures.MarkerQualityProcedure(acq)
        inspector = inspectFilters.QualityFilter(inspectprocedure2)
        inspector.run()




if __name__ == "__main__":

    Tests.test()
