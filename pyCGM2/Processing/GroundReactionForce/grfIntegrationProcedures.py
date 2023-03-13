import numpy as np
import scipy as sp



class GrfIntegrationProcedure(object):
    def __init__(self):
        pass


class gaitGrfIntegrationProcedure(GrfIntegrationProcedure):
    """


    """

    def __init__(self):
        """Procedure to cumulate the left and right time-normalized ground reaction forces measured during  gait 
        and extract the variation of the com velocities and positions (initial conditions set to 0) 

        """        
        super(gaitGrfIntegrationProcedure, self).__init__()
        
        self.centerOfmass = dict()
        self.centerOfmass["Left"] ={"Force": np.zeros((101,3)),"Acceleration": np.zeros((101,3)), "Velocity": np.zeros((101,3)),"Position": np.zeros((101,3))}
        self.centerOfmass["Right"] ={"Force": np.zeros((101,3)),"Acceleration": np.zeros((101,3)), "Velocity": np.zeros((101,3)),"Position": np.zeros((101,3))}

    def compute(self,analysisInstance):
        """compute the procedure

        Args:
            analysisInstance (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        """        

        # left
        values_L = np.nan_to_num(analysisInstance.kineticStats.data["LStanGroundReactionForce","Left"]["mean"])
        values_R = np.nan_to_num(analysisInstance.kineticStats.data["RStanGroundReactionForce","Right"]["mean"])        
        
        
        stanceL = analysisInstance.kineticStats.pst['stancePhase', "Left"]["mean"]
        stanceR = analysisInstance.kineticStats.pst['stancePhase', "Right"]["mean"]
        
        
        for i in range(0,101):
            self.centerOfmass["Left"]["Force"][i,:] =  values_L[i,:] + values_R[i+51,:] if i<50 else values_L[i,:]        
            self.centerOfmass["Right"]["Force"][i,:] =  values_R[i,:]+values_L[i+51,:] if i<50 else values_R[i,:] 

        self.centerOfmass["Left"]["Acceleration"] =  self.centerOfmass["Left"]["Force"]
        self.centerOfmass["Left"]["Acceleration"][:,2] =  self.centerOfmass["Left"]["Force"][:,2]-9.81

        self.centerOfmass["Right"]["Acceleration"] =  self.centerOfmass["Right"]["Force"]
        self.centerOfmass["Right"]["Acceleration"][:,2] =  self.centerOfmass["Right"]["Force"][:,2]-9.81


        for j in range(0,3):
            self.centerOfmass["Left"]["Velocity"][:,j] = sp.integrate.cumtrapz(self.centerOfmass["Left"]["Acceleration"][:,j],initial=0)
            self.centerOfmass["Left"]["Position"][:,j] = sp.integrate.cumtrapz(self.centerOfmass["Left"]["Velocity"][:,j], initial=0)

            self.centerOfmass["Right"]["Velocity"][:,j] = sp.integrate.cumtrapz(self.centerOfmass["Right"]["Acceleration"][:,j],initial=0)
            self.centerOfmass["Right"]["Position"][:,j] = sp.integrate.cumtrapz(self.centerOfmass["Right"]["Velocity"][:,j], initial=0)


        for i in range(0,101):
            if i > round(stanceL):
                self.centerOfmass["Left"]["Force"][i,:]=np.array([np.nan,np.nan,np.nan] )
                self.centerOfmass["Left"]["Velocity"][i,:]=np.array([np.nan,np.nan,np.nan] ) 
                self.centerOfmass["Left"]["Position"][i,:]=np.array([np.nan,np.nan,np.nan] )  
        
        for i in range(0,101):
            if i > round(stanceR):
                self.centerOfmass["Right"]["Force"][i,:]=np.array([np.nan,np.nan,np.nan] )
                self.centerOfmass["Right"]["Velocity"][i,:]=np.array([np.nan,np.nan,np.nan] ) 
                self.centerOfmass["Right"]["Position"][i,:]=np.array([np.nan,np.nan,np.nan] )  