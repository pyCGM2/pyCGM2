# -*- coding: utf-8 -*-
import numpy as np

class UnithanCoActivationProcedure(object):
    """

    """

    def __init__(self):
        pass

    def run(self,emg1,emg2):


        out = list()
        for c1,c2 in zip(emg1,emg2): # iterate along column
            commonEmg=np.zeros(((101,1)))
            for i in range(0,101):
                commonEmg[i,:]=np.minimum(c1[i],c2[i])
            res=np.trapz(commonEmg,x=np.arange(0,101),axis=0)[0]
            out.append(res)

        return out

class FalconerCoActivationProcedure(object):
    """

    """

    def __init__(self):
        pass

    def run(self,emg1,emg2):


        out = list()
        for c1,c2 in zip(emg1,emg2): # iterate along column

            import matplotlib.pyplot as plt
            plt.plot(c1)
            plt.plot(c2,"r-")

            commonEmg=np.zeros(((101,1)))
            sumEmg = np.zeros(((101,1)))
            for i in range(0,101):
                commonEmg[i,:]=np.minimum(c1[i],c2[i])
                sumEmg[i,:]=c1[i]+c2[i]

            areaNum=np.trapz(commonEmg,x=np.arange(0,101),axis=0)[0]
            areaDen=np.trapz(sumEmg,x=np.arange(0,101),axis=0)[0]
            res = 2.0* areaNum /areaDen
            out.append(res)

            plt.show()
        return out
