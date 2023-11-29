# -*- coding: utf-8 -*-
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Tools import  btkTools,opensimTools
import os
import numpy as np
import btk
import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Model.Opensim.interface import opensimInterface
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from pyCGM2.Model.Opensim.interface.procedures.scaling.opensimScalingInterfaceProcedure import ScalingXmlProcedure
from pyCGM2.Model.Opensim.interface.procedures.inverseKinematics.opensimInverseKinematicsInterfaceProcedure import InverseKinematicXmlProcedure
from pyCGM2.Model.Opensim.interface.procedures.inverseDynamics.opensimInverseDynamicsInterfaceProcedure import InverseDynamicsXmlProcedure
from pyCGM2.Model.Opensim.interface.procedures.staticOptimisation.opensimStaticOptimizationInterfaceProcedure import StaticOptimisationXmlProcedure
from pyCGM2.Model.Opensim.interface.procedures.analysisReport.opensimAnalysesInterfaceProcedure import AnalysesXmlCgmProcedure

from typing import List, Tuple, Dict, Optional,Union,Any

# pyCGM2
try:
    import btk
except:
    try:
        from pyCGM2 import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system")

try:
    import opensim
except:
    try:
        from pyCGM2 import opensim4 as opensim
    except:
        LOGGER.logger.error("[pyCGM2] opensim not found on your system")

from typing import List, Tuple, Dict, Optional,Union,Any

class opensimInterfaceScalingFilter(object):
    """
    Filter interface for OpenSim model scaling.

    Args:
        procedure (ScalingXmlProcedure): The scaling procedure to be applied.
    """

    def __init__(self, procedure:ScalingXmlProcedure):
        self.m_procedure = procedure

    def run(self):
        """
        Executes the scaling procedure.
        """
        self.m_procedure.run()

    def getOsimName(self) -> str:
        """
        Retrieves the name of the OpenSim model.

        Returns:
            str: The name of the OpenSim model.
        """
        return self.m_procedure.m_osimModel_name

    def getOsim(self):
        """
        Retrieves the OpenSim model.

        Returns:
            opensim.Model: The OpenSim model object.
        """
        return self.m_procedure.m_osimModel




class opensimInterfaceInverseKinematicsFilter(object):
    """
    Filter interface for OpenSim inverse kinematics.

    Args:
        procedure (InverseKinematicXmlProcedure): The inverse kinematics procedure to be applied.
    """
    def __init__(self, procedure:InverseKinematicXmlProcedure):
        self.m_procedure = procedure
        self.m_acq = None

    def run(self):
        """
        Executes the inverse kinematics procedure.
        """
        self.m_procedure.run()
        self.m_acq = self.m_procedure.m_acq0

    def getAcq(self):
        """
        Retrieves the btk acquisition .

        Returns:
            btk.btkAcquisition:The BTK acquisition instance where the procedure results are stored.
        """
        return self.m_acq

    def pushFittedMarkersIntoAcquisition(self):
        """
        Updates the acquisition object with fitted marker data from the inverse kinematics results.
        """
        marker_location_filename = self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\"+ self.m_procedure.m_dynamicFile+"_ik_model_marker_locations.sto"
        if os.path.isfile(marker_location_filename):
            os.remove(marker_location_filename)
        os.rename(self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+ "\\_ik_model_marker_locations.sto",
                    marker_location_filename)

        marker_errors_filename = self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\"+ self.m_procedure.m_dynamicFile+"_ik_marker_errors.sto"
        if os.path.isfile(marker_errors_filename):
            os.remove(marker_errors_filename)
        os.rename(self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\_ik_marker_errors.sto",
                 marker_errors_filename)

        acqMotionFinal = btk.btkAcquisition.Clone(self.m_procedure.m_acq0)

        # TODO : worl with storage datframe instead of the opensim sorage instance

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\", self.m_procedure.m_dynamicFile+"_ik_model_marker_locations.sto")

        storageObject = opensim.Storage(self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\"+self.m_procedure.m_dynamicFile +"_ik_model_marker_locations.sto")
        for marker in self.m_procedure.m_weights.keys():
            if self.m_procedure.m_weights[marker] != 0:
                values =opensimTools.sto2pointValues(storageObject,marker,self.m_procedure.m_R_LAB_OSIM)
                btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" )
                modelled = acqMotionFinal.GetPoint(marker).GetValues()
                ff = acqMotionFinal.GetFirstFrame()

                try:
                    modelled[self.m_procedure.m_frameRange[0]-ff:self.m_procedure.m_frameRange[1]-ff+1,:] = values
                except ValueError:
                    modelled[self.m_procedure.m_frameRange[0]-ff:self.m_procedure.m_frameRange[1]-ff+2,:] = values # FIX - sometimes we come across incompatible size 
                btkTools.smartAppendPoint(acqMotionFinal,marker, modelled, desc= "kinematic fitting" ) # new acq with marker overwrited


        self.m_acq = acqMotionFinal


    def pushMotToAcq(self, osimConverter):
        """
        Pushes motion data into the acquisition object.

        Args:
            osimConverter: A converter for OpenSim data to the acquisition format.
        """

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\", self.m_procedure.m_dynamicFile+".mot")

        for jointIt in osimConverter["Angles"]:

            values = np.zeros(
                (self.m_acq.GetPointFrameNumber(), 3))

            osimlabel_X = osimConverter["Angles"][jointIt]["X"]
            osimlabel_Y = osimConverter["Angles"][jointIt]["Y"]
            osimlabel_Z = osimConverter["Angles"][jointIt]["Z"]

            serie_X = storageDataframe.getDataFrame()[osimlabel_X]
            serie_Y = storageDataframe.getDataFrame()[osimlabel_Y]
            serie_Z = storageDataframe.getDataFrame()[osimlabel_Z]

            values[:, 0] = [+1*x for x in serie_X.to_list()]
            values[:, 1] = [+1*x for x in serie_Y.to_list()]
            values[:, 2] = [+1*x for x in serie_Z.to_list()]

            btkTools.smartAppendPoint(self.m_acq, jointIt
                                      + "_osim", values, PointType="Angle", desc="opensim angle")


class opensimInterfaceInverseDynamicsFilter(object):
    """
    Filter interface for OpenSim inverse dynamics.

    Args:
        procedure (InverseDynamicsXmlProcedure): The inverse dynamics procedure to be applied.
    """
    def __init__(self, procedure:InverseDynamicsXmlProcedure):
        self.m_procedure = procedure

    def run(self):
        """
        Runs the defined procedure.
        """
        self.m_procedure.run()

    def getAcq(self):
        """
        Retrieves the btk acquisition .

        Returns:
            btk.btkAcquisition:The BTK acquisition instance where the procedure results are stored.
        """
        return self.m_procedure.m_acq

    def pushStoToAcq(self, bodymass:float, osimConverter:Dict):
        """
        Pushes STO file data to BTK acquisition.

        Args:
            bodymass (float): The body mass used in the calculations.
            osimConverter (dict): A dictionary containing OpenSim conversion data.

        """

        if self.m_procedure.m_resultsDir == "":
            path = self.m_procedure.m_DATA_PATH
        else:
            path = self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\"

        if self.m_procedure.m_modelVersion == "":
            filename = self.m_procedure.m_dynamicFile + "-inverse_dynamics.sto"
        else:
            filename = self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion+"-inverse_dynamics.sto"

        storageDataframe = opensimIO.OpensimDataFrame(path,filename)
        
        for jointIt in osimConverter["Moments"]:

            values = np.zeros(
                (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

            osimlabel_X = osimConverter["Moments"][jointIt]["X"]
            osimlabel_Y = osimConverter["Moments"][jointIt]["Y"]
            osimlabel_Z = osimConverter["Moments"][jointIt]["Z"]

            serie_X = storageDataframe.getDataFrame()[osimlabel_X]
            serie_Y = storageDataframe.getDataFrame()[osimlabel_Y]
            serie_Z = storageDataframe.getDataFrame()[osimlabel_Z]

            values[:, 0] = [+1*x*1000/bodymass for x in serie_X.to_list()]
            values[:, 1] = [+1*x*1000/bodymass for x in serie_Y.to_list()]
            values[:, 2] = [+1*x*1000/bodymass for x in serie_Z.to_list()]

            btkTools.smartAppendPoint(self.m_procedure.m_acq, jointIt+"_osim",
                                      values, PointType="Moment", desc="opensim moment")


class opensimInterfaceStaticOptimizationFilter(object):
    """
    Filter interface for OpenSim static Optimisation.

    Args:
        procedure (StaticOptimisationXmlProcedure): The static optimisation procedure to be applied.
    """
    def __init__(self, procedure:StaticOptimisationXmlProcedure):
        self.m_procedure = procedure

    def run(self):
        """
        Runs the defined procedure.
        """
        self.m_procedure.run()

    def getAcq(self):
        """
        Retrieves the btk acquisition .

        Returns:
            btk.btkAcquisition:The BTK acquisition instance where the procedure results are stored.
        """
        return self.m_procedure.m_acq

    def pushStoToAcq(self):
        """
        Pushes STO file data to BTK acquisition.
        """    
        if self.m_procedure.m_modelVersion == "":
            filename = self.m_procedure.m_dynamicFile+"-analyses_StaticOptimization_force.sto"
        else:
            filename = self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_StaticOptimization_force.sto"

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\",filename)

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[StaticOptForce]", values, PointType="Scalar", desc="StaticOptForce")

        if self.m_procedure.m_modelVersion == "":
            filename = self.m_procedure.m_dynamicFile+"-analyses_StaticOptimization_activation.sto"
        else:
            filename = self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_StaticOptimization_activation.sto"
 
        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\",filename)

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[StaticOptActivation]", values, PointType="Scalar", desc="StaticOptActivation")


class opensimInterfaceAnalysesFilter(object):
    """
    Filter interface for OpenSim Analysis Report.

    Args:
        procedure (AnalysesXmlCgmProcedure): The analysis report procedure to be applied.
    """
    def __init__(self, procedure:AnalysesXmlCgmProcedure):
        self.m_procedure = procedure

        self.m_analysisType = None


    def run(self):
        """
        Runs the defined procedure.
        """
        self.m_procedure.run()

    def getAcq(self):
        """
        Retrieves the BTK acquisition instance where the procedure results are stored.

        Returns:
            Optional[btk.btkAcquisition]: The BTK acquisition instance, if available.
        """
        if self.m_procedure.m_acq is not None:
            return self.m_procedure.m_acq
        else:
            LOGGER.logger.warning("There is no Acquisition to return")


    def pushStoToAcq(self, type: str = "MuscleAnalysis", outputs: List[str] = ["Length"]) -> None:
        """
        Pushes STO file data to BTK acquisition.

        Args:
            type (str): The type of analysis to be processed (e.g., "MuscleAnalysis").
            outputs (List[str]): A list of outputs to be processed. The available options for "MuscleAnalysis" are:
                - MuscleActuatorPower
                - NormalizedFiberLength
                - NormFiberVelocity
                - PassiveFiberForce
                - PassiveFiberForceAlongTendon
                - PennationAngle
                - PennationAngularVelocity
                - TendonForce
                - TendonLength
                - TendonPower
                - ActiveFiberForce
                - ActiveFiberForceAlongTendon
                - FiberActivePower
                - FiberForce
                - FiberLength
                - FiberPassivePower
                - FiberVelocity
                - Length

        Raises:
            ValueError: If the acquisition is not available or if the specified 'type' or 'outputs' are invalid.
        """

         
        if self.m_procedure.m_acq is not None:
            
            nframes = self.m_procedure.m_acq.GetPointFrameNumber()

            for output in outputs:

                label = type[:-8] + output

                if self.m_procedure.m_modelVersion == "":
                    filename = self.m_procedure.m_dynamicFile+"-analyses_"+type+"_"+output+".sto"
                else:
                    filename = self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_"+type+"_"+output+".sto"

                storageDataframe = opensimIO.OpensimDataFrame(
                    self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\",
                    filename)

                values = np.zeros(
                    (self.m_procedure.m_acq.GetPointFrameNumber(), 3))
                
                freq = self.m_procedure.m_acq.GetPointFrequency()

                for muscle in storageDataframe.m_dataframe.columns[1:]:
                    serie = storageDataframe.getDataFrame()[muscle].to_list()
                    if freq !=100.0:
                        time = np.arange(0, len(serie)*(1/100), 1/100)
                        f = interp1d(time, serie, fill_value="extrapolate")
                        newTime = np.arange(0, len(serie)*(1/100), 1/freq)
                        values_interp = f(newTime)
                        
                        if values_interp.shape[0]>values.shape[0]:
                            values[:, 0] = values_interp[0:values.shape[0]]
                        else:
                            values[:, 0] = values_interp
                            
                    else:
                        values[:, 0] = serie


                    btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                            + "["+label+"]", values, PointType="Scalar", desc=label)
        
        else:
            LOGGER.logger.warning("There is no Acquisition. Nothing to push into")
