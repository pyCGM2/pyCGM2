import numpy as np
import scipy as sp
import pyCGM2
LOGGER = pyCGM2.LOGGER

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


    def compute(self,analysisInstance,bodymass):
        """compute the procedure

        Args:
            analysisInstance (pyCGM2.Processing.analysis.Analysis): an `analysis` instance
        """        
        leftFlag = False
        rightFlag = False

        for keyIt in  analysisInstance.kineticStats.data.keys():
            if keyIt[0] == "LStanGroundReactionForce" and  keyIt[1] == "Left":
                values_L = np.nan_to_num(analysisInstance.kineticStats.data["LStanGroundReactionForce","Left"]["mean"])*bodymass
                leftFlag = True
            if keyIt[0] == "RStanGroundReactionForce" and  keyIt[1] == "Right":
                values_R = np.nan_to_num(analysisInstance.kineticStats.data["RStanGroundReactionForce","Right"]["mean"])*bodymass        
                rightFlag = True
        
        if leftFlag and rightFlag:     
            stanceL = analysisInstance.kineticStats.pst['stancePhase', "Left"]["mean"]
            stanceR = analysisInstance.kineticStats.pst['stancePhase', "Right"]["mean"]
            stanceDurationL = analysisInstance.kineticStats.pst['stanceDuration', "Left"]["mean"]
            stanceDurationR = analysisInstance.kineticStats.pst['stanceDuration', "Right"]["mean"]
            

            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"] = {"Force": np.zeros((101,3)),"Acceleration": np.zeros((101,3)), "Velocity": np.zeros((101,3)),"Position": np.zeros((101,3))}
            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"] = {"Force": np.zeros((101,3)),"Acceleration": np.zeros((101,3)), "Velocity": np.zeros((101,3)),"Position": np.zeros((101,3))}



            for i in range(0,101):
                analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"][i,:] =  values_L[i,:] + values_R[i+51,:] if i<50 else values_L[i,:]        
                analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"][i,:] =  values_R[i,:]+values_L[i+51,:] if i<50 else values_R[i,:] 

            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"] =  (analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"])/bodymass
            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"][:,2] =  (analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Force"][:,2]-bodymass*9.81)/bodymass

            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"] =  (analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"])/bodymass
            analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"][:,2] =  (analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Force"][:,2]-bodymass*9.81)/bodymass


            for j in range(0,3):
                analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"][:,j] = sp.integrate.cumtrapz( \
                    analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Acceleration"][:,j],
                    dx = stanceDurationL/round(stanceL), initial=0)

                analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Position"][:,j] = sp.integrate.cumtrapz( \
                    analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"]["Velocity"][:,j],
                    dx = stanceDurationL/round(stanceL),initial=0)


                analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"][:,j] = sp.integrate.cumtrapz( \
                    analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Acceleration"][:,j],
                    dx = stanceDurationR/round(stanceR),initial=0)

                analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Position"][:,j] = sp.integrate.cumtrapz( \
                    analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"]["Velocity"][:,j],
                    dx = stanceDurationR/round(stanceR),initial=0)


            for i in range(0,101):
                if i > round(stanceL):
                    for key in analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"].keys():
                        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Left"][key][i,:]=np.array([np.nan,np.nan,np.nan] )

            
            for i in range(0,101):
                if i > round(stanceR):
                    for key in analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"].keys():
                        analysisInstance.kineticStats.optionalData["GaitNormalizedGRFIntegration","Right"][key][i,:]=np.array([np.nan,np.nan,np.nan] )
        else:
            LOGGER.logger.warning("[pyCGM2] - GRF Integration impossible - No left and Right Standardized GroundReactionForce")