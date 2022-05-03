# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Math
#APIDOC["Draft"]=False
#--end--

"""
Module contains function for working with euler angles

"""
import numpy as np
import copy


def wrapEulerTo(inputAngles, Dest):
    """correct euler angle

    Args:
        inputAngles (float): current angle value
        Dest (float): targeted value


    **Examples**

    ```python
        # extract from ModelJCSFilter
        dest = np.deg2rad(np.array([0,0,0]))
        for i in range (0, self.m_aqui.GetPointFrameNumber()):
            jointFinalValues[i,:] = euler.wrapEulerTo(np.deg2rad(jointFinalValues[i,:]), dest)
            dest = jointFinalValues[i,:]
    ```

    """

    an = copy.copy(inputAngles)
    bn = copy.copy(inputAngles)

    d1, an = _FixEuler(Dest, an)

    bn = bn * np.array([1, -1, 1]) + np.pi
    f1, bn = _FixEuler(Dest, bn)

    if f1 < d1:
        OutputAngles = bn
    else:
        OutputAngles = an

    return OutputAngles


def _FixEuler(Dest, Curr):

    Changed = Curr + np.pi*2 * np.floor((Dest - Curr + np.pi)/(np.pi*2))
    Distance = np.max(abs(Dest - Changed))

    return Distance, Changed


def _safeArcsin(Value):
    if np.abs(Value) > 1:
        Value = np.max(np.array([np.min(np.array([Value, 1])), -1]))

    return np.arcsin(Value)


def euler_xyz(Matrix, similarOrder=True):
    """
    Decomposition of a rotation matrix according the sequence XYZ.

    The function returns the 3 angles around X,Y,Z. ( the argument `similarOrder` has no effect for this sequence )

    Args:
        Matrix` (array[3,3]) - Rotation matrix
        similarOrder` (bool,Optional) -


    """
    # python translation  of the matlab pig .

    Euler2 = _safeArcsin(Matrix[0, 2])
    if(np.abs(np.cos(Euler2)) > np.spacing(np.single(1))*10):
      Euler1 = np.arctan2(-Matrix[1, 2], Matrix[2, 2])
      Euler3 = np.arctan2(-Matrix[0, 1], Matrix[0, 0])
    else:
      if(Euler2 > 0):
        Euler1 = np.arctan2(Matrix[1, 0], Matrix[1, 1])
      else:
        Euler1 = -np.arctan2(Matrix[0, 1], Matrix[1, 1])
      Euler3 = 0

    if similarOrder:
        return Euler1, Euler2, Euler3
    else:
        return Euler1, Euler2, Euler3


def euler_xzy(Matrix, similarOrder=True):
    """
    Decomposition of a rotation matrix according the sequence XZY.

    The function returns the 3 angles around X,Y,Z.
    If the argument `similarOrder` is enable, the returned angles map the angle sequence XZY
    ( ie 1st ouput: angle around X, 2nd ouput: angle around Z, 3rd ouput: angle around Y)

    Args:
        Matrix` (array[3,3]) - Rotation matrix
        similarOrder` (bool,Optional) - flag to return ouputs mapping the sequence ( default to True)
    """

    Euler3 = _safeArcsin(-Matrix[0, 1])
    if(np.abs(np.cos(Euler3)) > np.spacing(np.single(1))*10):
      Euler1 = np.arctan2(Matrix[2, 1], Matrix[1, 1])
      Euler2 = np.arctan2(Matrix[0, 2], Matrix[0, 0])
    else:
      if(Euler3 > 0):
        Euler1 = np.arctan2(-Matrix[2, 0], Matrix[2, 2])
      else:
        Euler1 = -np.arctan2(-Matrix[2, 0], Matrix[2, 2])
      Euler2 = 0

    if similarOrder:
        return Euler1, Euler3, Euler2
    else:
        return Euler1, Euler2, Euler3


def euler_yxz(Matrix, similarOrder=True):
    """
    Decomposition of a rotation matrix according the sequence YXZ.

    The function returns the 3 angles around X,Y,Z.
    If the argument `similarOrder` is enable, the returned angles map the angle sequence YXZ
    ( ie 1st ouput: angle around Y, 2nd ouput: angle around X, 3rd ouput: angle around Z)

    Args:
        Matrix` (array[3,3]) - Rotation matrix
        similarOrder` (bool,Optional) - flag to return ouputs mapping the sequence ( default to True)
    """

    Euler1 = _safeArcsin(-Matrix[1, 2])
    if(np.abs(np.cos(Euler1)) > np.spacing(np.single(1))*10):
      Euler2 = np.arctan2(Matrix[0, 2], Matrix[2, 2])
      Euler3 = np.arctan2(Matrix[1, 0], Matrix[1, 1])
    else:
      if(Euler1 > 0):
        Euler2 = np.arctan2(-Matrix[0, 1], Matrix[0, 0])
      else:
        Euler2 = -np.arctan2(-Matrix[0, 1], Matrix[0, 0])
      Euler3 = 0

    if similarOrder:
        return Euler2, Euler1, Euler3
    else:
        return Euler1, Euler2, Euler3


def euler_yzx(Matrix, similarOrder=True):
    """
    Decomposition of a rotation matrix according the sequence YZX.

    The function returns the 3 angles around X,Y,Z.
    If the argument `similarOrder` is enable, the returned angles map the angle sequence YZX
    ( ie 1st ouput: angle around Y, 2nd ouput: angle around Z, 3rd ouput: angle around X)

    Args:
        Matrix` (array[3,3]) - Rotation matrix
        similarOrder` (bool,Optional) - flag to return ouputs mapping the sequence ( default to True)
    """

    Euler3 = _safeArcsin(Matrix[1, 0])
    if(np.abs(np.cos(Euler3)) > np.spacing(np.single(1))*10):
      Euler1 = np.arctan2(-Matrix[1, 2], Matrix[1, 1])
      Euler2 = np.arctan2(-Matrix[2, 0], Matrix[0, 0])
    else:
      if(Euler3 > 0):
        Euler2 = np.arctan2(Matrix[2, 1], Matrix[2, 2])
      else:
        Euler2 = -np.arctan2(Matrix[2, 1], Matrix[2, 2])
      Euler1 = 0

    if similarOrder:
        return Euler2, Euler3, Euler1
    else:
        return Euler1, Euler2, Euler3


def euler_zxy(Matrix, similarOrder=True):
    """
    Decomposition of a rotation matrix according the sequence ZXY.

    The function returns the 3 angles around X,Y,Z.
    If the argument `similarOrder` is enable, the returned angles map the angle sequence ZXY
    ( ie 1st ouput: angle around Z, 2nd ouput: angle around X, 3rd ouput: angle around Y)

    Args:
        Matrix` (array[3,3]) - Rotation matrix
        similarOrder` (bool,Optional) - flag to return ouputs mapping the sequence ( default to True)
    """

    Euler1 = _safeArcsin(Matrix[2, 1])
    if(np.abs(np.cos(Euler1)) > np.spacing(np.single(1))*10):
      Euler2 = np.arctan2(-Matrix[2, 0], Matrix[2, 2])
      Euler3 = np.arctan2(-Matrix[0, 1], Matrix[1, 1])
    else:
      if(Euler1 > 0):
        Euler3 = np.arctan2(Matrix[0, 2], Matrix[0, 0])
      else:
        Euler3 = -np.arctan2(Matrix[0, 2], Matrix[0, 0])
      Euler2 = 0

    if similarOrder:
        return Euler3, Euler1, Euler2
    else:
        return Euler1, Euler2, Euler3


def euler_zyx(Matrix, similarOrder=True):
    """
    Decomposition of a rotation matrix according the sequence ZYX.

    The function returns the 3 angles around X,Y,Z.
    If the argument `similarOrder` is enable, the returned angles map the angle sequence ZYX
    ( ie 1st ouput: angle around Z, 2nd ouput: angle around Y, 3rd ouput: angle around X)

    Args:
        Matrix` (array[3,3]) - Rotation matrix
        similarOrder` (bool,Optional) - flag to return ouputs mapping the sequence ( default to True)
    """

    Euler2 = _safeArcsin(-Matrix[2, 0])
    if(np.abs(np.cos(Euler2)) > np.spacing(np.single(1))*10):
      Euler1 = np.arctan2(Matrix[2, 1], Matrix[2, 2])
      Euler3 = np.arctan2(Matrix[1, 0], Matrix[0, 0])
    else:
      if(Euler2 > 0):
        Euler3 = np.arctan2(-Matrix[0, 1], Matrix[0, 2])
      else:
        Euler3 = -np.arctan2(-Matrix[0, 1], Matrix[0, 2])
      Euler1 = 0

    if similarOrder:
        return Euler3, Euler2, Euler1
    else:
        return Euler1, Euler2, Euler3
