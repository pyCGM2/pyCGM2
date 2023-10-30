# -*- coding: utf-8 -*-
import numpy as np
from pyCGM2.Math import pose


class ImuReaderFilter(object):
    def __init__(self,procedure):
        self.m_procedure=procedure
  
    def run(self):
        return self.m_procedure.read()

class ImuMotionFilter(object):
    def __init__(self,imuInstance,procedure):
        self.m_imu=imuInstance
        self.m_procedure=procedure
  
    def run(self):
        self.m_procedure.compute(self.m_imu)




class ImuRelativeAnglesFilter(object):
    def __init__(self,imuInstance1,imuInstance2,procedure):
        self.m_imu1=imuInstance1
        self.m_imu2=imuInstance2
        self.m_procedure=procedure

    def run(self):
        out = self.m_procedure.run(self.m_imu1,self.m_imu2)
        return out