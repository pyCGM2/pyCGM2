# -*- coding: utf-8 -*-
import numpy as np

def safeArcsin( Value):
    if np.abs( Value ) > 1:
        Value = np.max(  np.array( [ np.min( np.array([Value,1]) ), -1 ] ))
  
    return np.arcsin(Value)


def euler_xyz(Matrix, similarOrder = True):
    """ 
        Decomposition of a rotation matrix according the sequence XYZ
        
        :Parameters:
           - `Matrix` (numpy.array(3,3)) - Rotation matrix
           - `similarOrder` (bool) - return in same order than sequence   
        
        :Return:
            - `euler1` (float) - angle for X-axis
            - `euler2` (float) - angle for Y-axis
            - `euler3` (float) - angle for Z-axis
    """
    # python translation  of the matlab pig .  

    
    Euler2= safeArcsin( Matrix[0,2] )
    if( np.abs( np.cos( Euler2 ) ) > np.spacing(np.single(1))*10 ):
      Euler1 = np.arctan2( -Matrix[1,2], Matrix[2,2] )
      Euler3 = np.arctan2( -Matrix[0,1], Matrix[0,0] )
    else:
      if( Euler2 > 0 ):
        Euler1 = np.arctan2( Matrix[1,0], Matrix[1,1] )
      else:
        Euler1 = -np.arctan2( Matrix[0,1], Matrix[1,1] )
      Euler3 = 0

    if similarOrder:
        return Euler1,Euler2,Euler3
    else:
        return Euler1,Euler2,Euler3


def euler_xzy(Matrix, similarOrder = True):
    """ 
        Decomposition of a rotation matrix according the sequence XZY
        
        :Parameters:
           - `Matrix` (numpy.array(3,3)) : Rotation matrix
           - `similarOrder` (bool) : return in same order than sequence   
        
        :Return:
            - `euler1` (float) - angle for X-axis
            - `euler2` (float) - angle for Y-axis
            - `euler3` (float) - angle for Z-axis
    """

    
    Euler3= safeArcsin( -Matrix[0,1] )
    if( np.abs( np.cos( Euler3 ) ) > np.spacing(np.single(1))*10 ):
      Euler1 = np.arctan2( Matrix[2,1], Matrix[1,1] )
      Euler2 = np.arctan2( Matrix[0,2], Matrix[0,0] )
    else:
      if( Euler3 > 0 ):
        Euler1 = np.arctan2( -Matrix[2,0], Matrix[2,2] )
      else:
        Euler1 = -np.arctan2( -Matrix[2,0], Matrix[2,2] )
      Euler2 = 0

    if similarOrder:
        return Euler1,Euler3,Euler2
    else:
        return Euler1,Euler2,Euler3




def euler_yxz(Matrix, similarOrder = True ):
    """ 
        Decomposition of a rotation matrix according the sequence YXZ
        
        :Parameters:
           - `Matrix` (numpy.array(3,3)) : Rotation matrix
           - `similarOrder` (bool) : return in same order than sequence   
        
        :Return:
            - `euler1` (float) - angle for X-axis
            - `euler2` (float) - angle for Y-axis
            - `euler3` (float) - angle for Z-axis
    """     
    
    
    
    Euler1= safeArcsin( -Matrix[1,2] )
    if( np.abs( np.cos( Euler1 ) ) > np.spacing(np.single(1))*10 ):
      Euler2 = np.arctan2( Matrix[0,2], Matrix[2,2] )
      Euler3 = np.arctan2( Matrix[1,0], Matrix[1,1] )
    else:
      if( Euler1 > 0 ):
        Euler2 = np.arctan2( -Matrix[0,1], Matrix[0,0] )
      else:
        Euler2 = -np.arctan2( -Matrix[0,1], Matrix[0,0] )
      Euler3 = 0

    if similarOrder:
        return Euler2,Euler1,Euler3
    else:
        return Euler1,Euler2,Euler3

def euler_yzx(Matrix, similarOrder = True):
    """ 
        Decomposition of a rotation matrix according the sequence YZX
        
        :Parameters:
           - `Matrix` (numpy.array(3,3)) : Rotation matrix
           - `similarOrder` (bool) : return in same order than sequence   
        
        :Return:
            - `euler1` (float) - angle for X-axis
            - `euler2` (float) - angle for Y-axis
            - `euler3` (float) - angle for Z-axis
    """            
    
    
    Euler3= safeArcsin( Matrix[1,0] )
    if( np.abs( np.cos( Euler3 ) ) > np.spacing(np.single(1))*10 ):
      Euler1 = np.arctan2( -Matrix[1,2], Matrix[1,1] )
      Euler2 = np.arctan2( -Matrix[2,0], Matrix[0,0] )
    else:
      if( Euler3 > 0 ):
        Euler2 = np.arctan2( Matrix[2,1], Matrix[2,2] )
      else:
        Euler2 = -np.arctan2( Matrix[2,1], Matrix[2,2] )
      Euler1 = 0

    if similarOrder:
        return Euler2,Euler3,Euler1
    else:
        return Euler1,Euler2,Euler3


def euler_zxy(Matrix, similarOrder = True):
    """ 
        Decomposition of a rotation matrix according the sequence ZXY
        
        :Parameters:
           - `Matrix` (numpy.array(3,3)) : Rotation matrix
           - `similarOrder` (bool) : return in same order than sequence   
        
        :Return:
            - `euler1` (float) - angle for X-axis
            - `euler2` (float) - angle for Y-axis
            - `euler3` (float) - angle for Z-axis
    """  

    
    Euler1= safeArcsin( Matrix[2,1] )
    if( np.abs( np.cos( Euler1 ) ) > np.spacing(np.single(1))*10 ):
      Euler2 = np.arctan2( -Matrix[2,0], Matrix[2,2] )
      Euler3 = np.arctan2( -Matrix[0,1], Matrix[1,1] )
    else:
      if( Euler1 > 0 ):
        Euler3 = np.arctan2( Matrix[0,2], Matrix[0,0] )
      else:
        Euler3 = -np.arctan2( Matrix[0,2], Matrix[0,0] )
      Euler2 = 0

    if similarOrder:
        return Euler3,Euler1,Euler2
    else:
        return Euler1,Euler2,Euler3


def euler_zyx(Matrix, similarOrder = True):
    """ 
        Decomposition of a rotation matrix according the sequence ZYX
        
        :Parameters:
           - `Matrix` (numpy.array(3,3)) : Rotation matrix
           - `similarOrder` (bool) : return in same order than sequence   
        
        :Return:
            - `euler1` (float) - angle for X-axis
            - `euler2` (float) - angle for Y-axis
            - `euler3` (float) - angle for Z-axis
    """  

    
    Euler2= safeArcsin( -Matrix[2,0] )
    if( np.abs( np.cos( Euler2 ) ) > np.spacing(np.single(1))*10 ):
      Euler1 = np.arctan2( Matrix[2,1], Matrix[2,2] )
      Euler3 = np.arctan2( Matrix[1,0], Matrix[0,0] )
    else:
      if( Euler2 > 0 ):
        Euler3 = np.arctan2( -Matrix[0,1], Matrix[0,2] )
      else:
        Euler3 = -np.arctan2( -Matrix[0,1], Matrix[0,2] )
      Euler1 = 0

    if similarOrder:
        return Euler3,Euler2,Euler1
    else:
        return Euler1,Euler2,Euler3    
    
    
    



    



    
    
    
