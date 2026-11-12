# -*- coding: utf-8 -*-
"""Anthropometric measurements computed from marker trajectories."""
import numpy as np

def measureNorm(acq,MarkerLabel1,MarkerLabel2,markerDiameter=0):
    """Measure the mean distance between two markers.

    Args:
        acq (btk.btkAcquisition): Acquisition.
        MarkerLabel1 (str): First marker label.
        MarkerLabel2 (str): Second marker label.
        markerDiameter (float, optional): Marker diameter subtracted from the distance. Defaults to 0.

    Returns:
        float: Mean distance.
    """
    mv1 = acq.GetPoint(MarkerLabel1).GetValues()
    mv2 = acq.GetPoint(MarkerLabel2).GetValues()

    norm = np.linalg.norm(mv1-mv2, axis=1)

    return norm.mean()-markerDiameter
