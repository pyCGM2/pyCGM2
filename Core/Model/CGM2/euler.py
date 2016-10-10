# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 11:14:18 2016

@author: aaa34169


TODO : complete documentation

"""

import numpy as np


def safeArcsin( Value):
    if np.abs( Value ) > 1:
        Value = np.max(  np.array( [ np.min( np.array([Value,1]) ), -1 ] ))
  
    return np.arcsin(Value)


def euler_xyz(Matrix, similarOrder = True):
    """ decomposition of a rotation matrix according the sequence XYZ
        
        :Parameters:
           - `Matrix` (array)(3,3) : Rotation Matrix
           - `similarOrder (bool) : return in same order than sequence   
        
        Returns:
            - euler1: angle for X-axis
            - euler2: angle for Y-axis
            - euler3: angle for Z-axis

        .. note:: python adaptation of the pig Matlab.  

    case EEulerOrder.EXYZ
    Euler(2) = SafeArcsin( Matrix(1,3) );
    if( abs( cos( Euler(2) ) ) > eps('single')*10 )
      Euler(1) = atan2( -Matrix(2,3), Matrix(3,3) );
      Euler(3) = atan2( -Matrix(1,2), Matrix(1,1) );
    else
      if( Euler(2) > 0 )
        Euler(1) = atan2( Matrix(2,1), Matrix(2,2) );
      else
        Euler(1) = -atan2( Matrix(1,2), Matrix(2,2) );
      end
      Euler(3) = 0;
    end    """
    


    
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
     case EEulerOrder.EXZY
    Euler(3) = SafeArcsin( -Matrix(1,2) );
    if( abs( cos( Euler(3) ) ) > eps('single')*10 )
      Euler(1) = atan2( Matrix(3,2), Matrix(2,2) );
      Euler(2) = atan2( Matrix(1,3), Matrix(1,1) );
    else
      if( Euler(3) > 0 )
        Euler(1) = atan2( -Matrix(3,1), Matrix(3,3) );
      else
        Euler(1) = -atan2( -Matrix(3,1), Matrix(3,3) );
      end
      Euler(2) = 0;
    end
    
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
    
    
          case EEulerOrder.EYXZ
    Euler(1) = SafeArcsin( -Matrix(2,3) );
    if( abs( cos( Euler(1) ) ) > eps('single')*10 )
      Euler(2) = atan2( Matrix(1,3), Matrix(3,3) );
      Euler(3) = atan2( Matrix(2,1), Matrix(2,2) );
    else
      if( Euler(1) > 0 )
        Euler(2) = atan2( -Matrix(1,2), Matrix(1,1) );
      else
        Euler(2) = -atan2( -Matrix(1,2), Matrix(1,1) );
      end
      Euler(3) = 0;
    end  
    
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
    """ case EEulerOrder.EYZX
    Euler(3) = SafeArcsin( Matrix(2,1) );
    if( abs( cos( Euler(3) ) ) > eps('single')*10 )
      Euler(1) = atan2( -Matrix(2,3), Matrix(2,2) );
      Euler(2) = atan2( -Matrix(3,1), Matrix(1,1) );
    else
      if( Euler(3) > 0 )
        Euler(2) = atan2( Matrix(3,2), Matrix(3,3) );
      else
        Euler(2) = -atan2( Matrix(3,2), Matrix(3,3) );
      end
      Euler(1) = 0;
    end


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
   case EEulerOrder.EZXY
    Euler(1) = SafeArcsin( Matrix(3,2) );
    if( abs( cos( Euler(1) ) ) > eps('single')*10 )
      Euler(2) = atan2( -Matrix(3,1), Matrix(3,3) );
      Euler(3) = atan2( -Matrix(1,2), Matrix(2,2) );
    else
      if( Euler(1) > 0 )
        Euler(3) = atan2( Matrix(1,3), Matrix(1,1) );
      else
        Euler(3) = -atan2( Matrix(1,3), Matrix(1,1) );
      end
      Euler(2) = 0;
    end
    
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
     case EEulerOrder.EZYX
    Euler(2) = SafeArcsin( -Matrix(3,1) );
    if( abs( cos( Euler(2) ) ) > eps('single')*10 )
      Euler(1) = atan2( Matrix(3,2), Matrix(3,3) );
      Euler(3) = atan2( Matrix(2,1), Matrix(1,1) );
    else
      if( Euler(2) > 0 )
        Euler(3) = atan2( -Matrix(1,2), Matrix(1,3) );
      else
        Euler(3) = -atan2( -Matrix(1,2), Matrix(1,3) );
      end
      Euler(1) = 0;
    end

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
    
    
    



    



    
    
    
