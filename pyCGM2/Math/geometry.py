import numpy as np
from typing import List, Tuple, Dict, Optional,Union

def computeAngle(u1:np.ndarray,v1:np.ndarray)->float:
    """
    Compute the angle between two vectors.

    Args:
        u1 (np.ndarray): First vector.
        v1 (np.ndarray): Second vector.

    Returns:
        float: Angle in radians between the two vectors.
    """
    if len(u1)==3:
    #     %3D, can use cross to resolve sign
        uMod = np.linalg.norm(u1)
        vMod = np.linalg.norm(v1)
        uvPr = np.sum(u1*v1)
        costheta = min(uvPr/uMod/vMod,1)

        theta = np.arccos(costheta)
    #
    #     %resolve sign
        cp=(np.cross(u1,v1))
        idxM = np.argmax(abs(cp)) #idxM=find(abs(cp)==max(abs(cp)));

        s= cp[idxM]
        if s < 0:
            theta = -theta;
    elif len(u1)==2:
        theta = (np.arctan2(v1[1],v1[0])-np.arctan2(u1[1],u1[0]))

    return theta

def angleFrom2Vectors(v1:np.ndarray, v2:np.ndarray, vn:Optional[np.ndarray]=None)->float:
    """
    Compute a signed angle between two vectors, optionally using a common orthogonal vector.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        vn (Optional[np.ndarray], optional): Common orthogonal vector. Defaults to None.

    Returns:
        float: Signed angle between the two vectors.
    """

    cross = np.cross(v1, v2)
    angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v2, v1))

    # if vn is not None:
    #     if (np.dot(vn, cross) < 0):
    #         angle = -angle;

    return angle


def oppositeVector(v1:np.ndarray)->np.ndarray:
    """
    Compute the opposite (negative) of a vector.

    Args:
        v1 (np.ndarray): Input vector.

    Returns:
        np.ndarray: Opposite vector.
    """
    return (-1*np.ones(3))*v1


def LineLineIntersect(p1:np.ndarray, p2:np.ndarray, p3:np.ndarray, p4:np.ndarray):
    """
    Calculates the line segment pa_pb that is the shortest route
    between two lines p1_p2 and p3_p4. Calculates also the values of
    mua and mub where:

      - pa = p1 + mua (p2 - p1)
      - pb = p3 + mub (p4 - p3)

    Args:
        p1 (np.ndarray): Start point of the first line.
        p2 (np.ndarray): End point of the first line.
        p3 (np.ndarray): Start point of the second line.
        p4 (np.ndarray): End point of the second line.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The points pa and pb on the first and second line, respectively,
                                      where the shortest line segment intersects each line.
    
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
