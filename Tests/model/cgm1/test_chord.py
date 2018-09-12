# -*- coding: utf-8 -*-
import numpy as np
import pdb
import logging

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)


# pyCGM2
from pyCGM2 import enums
from pyCGM2.Tools import  btkTools
from pyCGM2.Model import  modelFilters,modelDecorator, frame, modelQualityFilter
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Utils import timer

class chordTests():

    @classmethod
    def test0(cls):

        offset = 67.0
        beta = -8.006

        value = np.load("test.out.npy")

        pt1 = value[0]
        pt2 = value[1]
        pt3 = value[2]

        with timer.Timer("usage"):
            val0 = modelDecorator.chord (offset,pt1,pt2,pt3,beta=beta)

        with timer.Timer("usage2"):
            val1 = modelDecorator.chord (offset,pt1,pt2,pt3,beta=beta,epsilon = 0.01)

        import ipdb
        ipdb.set_trace()


if __name__ == "__main__":
    chordTests.test0()

    #FraserAcq.test1()
