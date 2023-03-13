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

        values_L = np.nan_to_num(analysisInstance.kineticStats.data["LStanGroundReactionForce","Left"]["mean"])
        values_R = np.nan_to_num(analysisInstance.kineticStats.data["RStanGroundReactionForce","Right"]["mean"])        
        
        
        stanceL = analysisInstance.kineticStats.pst['stancePhase', "Left"]["mean"]
        stanceR = analysisInstance.kineticStats.pst['stancePhase', "Right"]["mean"]
        

        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"] = {"Force": np.zeros((101,3)),"Acceleration": np.zeros((101,3)), "Velocity": np.zeros((101,3)),"Position": np.zeros((101,3))}
        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"] = {"Force": np.zeros((101,3)),"Acceleration": np.zeros((101,3)), "Velocity": np.zeros((101,3)),"Position": np.zeros((101,3))}



        for i in range(0,101):
            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"][i,:] =  values_L[i,:] + values_R[i+51,:] if i<50 else values_L[i,:]        
            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"][i,:] =  values_R[i,:]+values_L[i+51,:] if i<50 else values_R[i,:] 

        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"] =  analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"]
        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"][:,2] =  analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"][:,2]-9.81

        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"] =  analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"]
        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"][:,2] =  analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"][:,2]-9.81


        for j in range(0,3):
            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"][:,j] = sp.integrate.cumtrapz( \
                 analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"][:,j],initial=0)

            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Position"][:,j] = sp.integrate.cumtrapz( \
                 analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"][:,j],initial=0)


            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"][:,j] = sp.integrate.cumtrapz( \
                 analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"][:,j],initial=0)

            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Position"][:,j] = sp.integrate.cumtrapz( \
                 analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"][:,j],initial=0)


        for i in range(0,101):
            if i > round(stanceL):
                for key in analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"].keys():
                    analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"][key][i,:]=np.array([np.nan,np.nan,np.nan] )

        
        for i in range(0,101):
            if i > round(stanceR):
                for key in analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"].keys():
                    analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"][key][i,:]=np.array([np.nan,np.nan,np.nan] )