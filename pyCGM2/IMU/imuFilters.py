# -*- coding: utf-8 -*-


# class ImuOrinetationFilter(object):
    

#     def __init__(self,imuInstance,procedure):
#         self.m_imu=imuInstance

#     def compute(self):
#         out = self.m_procedure.run(self.m_imu1)
#         return out



class RelativeIMUAnglesFilter(object):
    

    def __init__(self,imuInstance1,imuInstance2,procedure):
        self.m_imu1=imuInstance1
        self.m_imu2=imuInstance2
        self.m_procedure=procedure

    def compute(self):
        out = self.m_procedure.run(self.m_imu1,self.m_imu2)
        return out