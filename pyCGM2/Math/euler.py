"""
Module contains functions for working with euler angles

"""
import numpy as np
import copy

from typing import List, Tuple, Dict, Optional,Union

def wrapEulerTo(inputAngles:np.ndarray, Dest:np.ndarray):
    """
    Correct Euler angles to wrap around the target values.

    Args:
        inputAngles (np.ndarray): Current Euler angle values.
        Dest (np.ndarray): Targeted Euler angle values to wrap to.

    Returns:
        np.ndarray: Corrected Euler angles wrapped to the target values.
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
    """
    Internal function to fix Euler angles with respect to a destination angle.

    Args:
        Dest (np.ndarray): Destination Euler angles.
        Curr (np.ndarray): Current Euler angles to be adjusted.

    Returns:
        Tuple[float, np.ndarray]: The distance and adjusted Euler angles.
    """
    Changed = Curr + np.pi*2 * np.floor((Dest - Curr + np.pi)/(np.pi*2))
    Distance = np.max(abs(Dest - Changed))

    return Distance, Changed


def _safeArcsin(Value):
    """
    Safe computation of arcsin, ensuring the input value is within valid range.

    Args:
        Value (float): Input value for arcsin computation.

    Returns:
        float: Result of the arcsin computation.
    """
    if np.abs(Value) > 1:
        Value = np.max(np.array([np.min(np.array([Value, 1])), -1]))

    return np.arcsin(Value)


def euler_xyz(Matrix:np.ndarray, similarOrder:bool=True):
    """
    Decompose a rotation matrix using the XYZ Euler sequence.

    Args:
        Matrix (np.ndarray): Rotation matrix to decompose.
        similarOrder (bool, optional): Flag to return outputs mapping the XYZ sequence. Defaults to True.

    Returns:
        Tuple[float, float, float]: Euler angles (XYZ) derived from the rotation matrix.
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


def euler_xzy(Matrix:np.ndarray, similarOrder:bool=True):
    """
    Decompose a rotation matrix using the YZX Euler sequence.

    Args:
        Matrix (np.ndarray): Rotation matrix to decompose.
        similarOrder (bool, optional): Flag to return outputs mapping the YZX sequence. Defaults to True.

    Returns:
        Tuple[float, float, float]: Euler angles (YZX) derived from the rotation matrix.
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


def euler_yxz(Matrix:np.ndarray, similarOrder:bool=True):
    """
    Decompose a rotation matrix using the YXZ Euler sequence.

    Args:
        Matrix (np.ndarray): Rotation matrix to decompose.
        similarOrder (bool, optional): Flag to return outputs mapping the YXZ sequence. Defaults to True.

    Returns:
        Tuple[float, float, float]: Euler angles (YXZ) derived from the rotation matrix.
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


def euler_yzx(Matrix:np.ndarray, similarOrder:bool=True):
    """
    Decompose a rotation matrix using the ZYX Euler sequence.

    Args:
        Matrix (np.ndarray): Rotation matrix to decompose.
        similarOrder (bool, optional): Flag to return outputs mapping the ZYX sequence. Defaults to True.

    Returns:
        Tuple[float, float, float]: Euler angles (ZYX) derived from the rotation matrix.
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


def euler_zxy(Matrix:np.ndarray, similarOrder:bool=True):
    """
    Decompose a rotation matrix using the ZXY Euler sequence.

    Args:
        Matrix (np.ndarray): Rotation matrix to decompose.
        similarOrder (bool, optional): Flag to return outputs mapping the ZXY sequence. Defaults to True.

    Returns:
        Tuple[float, float, float]: Euler angles (ZXY) derived from the rotation matrix.
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


def euler_zyx(Matrix, similarOrder:bool=True):
    """
    Decompose a rotation matrix using the XZY Euler sequence.

    Args:
        Matrix (np.ndarray): Rotation matrix to decompose.
        similarOrder (bool, optional): Flag to return outputs mapping the XZY sequence. Defaults to True.

    Returns:
        Tuple[float, float, float]: Euler angles (XZY) derived from the rotation matrix.
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
