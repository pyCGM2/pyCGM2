# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Math
#APIDOC["Draft"]=False
#--end--

import numpy as np


def angleFrom2Vectors(v1, v2, vn=None):
    """
    Return a signed angle between 2 vectors.
    The common orthogonal vector is used for defining the sign of the angle

    Args:
        v1 (array[3,]): first vector
        v2 (array[3,]): second vector
        vn (array[3,]): common orthogonal vector.

    """

    cross = np.cross(v1, v2)
    angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v2, v1))

    # if vn is not None:
    #     if (np.dot(vn, cross) < 0):
    #         angle = -angle;

    return angle


def oppositeVector(v1):
    return (-1*np.ones(3))*v1


def LineLineIntersect(p1, p2, p3, p4):
    """
    Calculates the line segment pa_pb that is the shortest route
    between two lines p1_p2 and p3_p4. Calculates also the values of
    mua and mub where:

      - pa = p1 + mua (p2 - p1)
      - pb = p3 + mub (p4 - p3)

    Args:
        p1 (np.array(3)) : 3d coordinates
        p2 (np.array(3)) : 3d coordinates
        p3 (np.array(3)) : 3d coordinates
        p4 (np.array(3)) : 3d coordinates

    note::

        this a python conversion from the code proposed by
        Paul Bourke at http://astronomy.swin.edu.au/~pbourke/geometry/lineline3d/

    """
    pa = np.zeros((3))
    pb = np.zeros((3))

    p13 = np.zeros((3))
    p43 = np.zeros((3))

    p13[0] = p1[0] - p3[0]
    p13[1] = p1[1] - p3[1]
    p13[2] = p1[2] - p3[2]

    p43[0] = p4[0] - p3[0]
    p43[1] = p4[1] - p3[1]
    p43[2] = p4[2] - p3[2]

    if ((np.abs(p43[0]) < np.spacing(1)) and (np.abs(p43[1]) < np.spacing(1)) and (np.abs(p43[2]) < np.spacing(1))):
        raise Exception(" [pyCGM2] Could not compute LineLineIntersect! ")

    p21 = np.zeros((3))

    p21[0] = p2[0] - p1[0]
    p21[1] = p2[1] - p1[1]
    p21[2] = p2[2] - p1[2]

    if ((np.abs(p21[0]) < np.spacing(1)) and (np.abs(p21[1]) < np.spacing(1)) and (np.abs(p21[2]) < np.spacing(1))):
        raise Exception(" [pyCGM2] Could not compute LineLineIntersect! ")

    d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2]
    d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2]
    d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2]
    d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2]
    d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2]

    denom = d2121 * d4343 - d4321 * d4321

    if (np.abs(denom) < np.spacing(1)):
        raise Exception(" [pyCGM2] Could not compute LineLineIntersect! ")

    numer = d1343 * d4321 - d1321 * d4343

    mua = numer / denom
    mub = (d1343 + d4321 * mua) / d4343

    pa[0] = p1[0] + mua * p21[0]
    pa[1] = p1[1] + mua * p21[1]
    pa[2] = p1[2] + mua * p21[2]
    pb[0] = p3[0] + mub * p43[0]
    pb[1] = p3[1] + mub * p43[1]
    pb[2] = p3[2] + mub * p43[2]

    return pa, pb
