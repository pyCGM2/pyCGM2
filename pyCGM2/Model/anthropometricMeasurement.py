# -*- coding: utf-8 -*-
import numpy as np

def measureNorm(acq,MarkerLabel1,MarkerLabel2,markerDiameter=0):
    mv1 = acq.GetPoint(MarkerLabel1).GetValues()
    mv2 = acq.GetPoint(MarkerLabel2).GetValues()

    norm = np.linalg.norm(mv1-mv2, axis=1)

    return norm.mean()-markerDiameter
