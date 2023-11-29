

import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER

from pyCGM2.Model.model import Model
from typing import List, Tuple, Dict, Optional,Union,Any

def updateFromcustomMp(model:Model, custom_mp:Dict):
    """
    Update the anthropometric parameters of the model with custom measurements.

    Args:
        model (Model): An instance of a pyCGM2 Model.
        custom_mp (dict): Anthropometric parameters from the file mp.settings.
    """
    for segment in model.m_segmentCollection:
        name = segment.name
        if name in custom_mp["mp"]:
            mass = custom_mp["mp"][name]["mass"]
            length = custom_mp["mp"][name]["length"]
            com = custom_mp["mp"][name]["com"]
            rog = custom_mp["mp"][name]["rog"]
            inertia = custom_mp["mp"][name]["inertia"]

            if inertia is not None:
                if len(inertia) !=9:
                    LOGGER.logger.error("Incorrect dimensions of your inertia parameter. 9 elements required")
                    raise 
                segment.m_bsp["inertia"] = np.array(inertia).reshape((3,3))
                LOGGER.logger.info("default inertia of the segment [%s] updated"%(name))

            if mass is not None: 
                segment.m_bsp["mass"] = mass
                LOGGER.logger.info("default mass of the segment [%s] updated"%(name))

            if length is not None: 
                segment.m_bsp["length"] = length
                LOGGER.logger.info("default length of the segment [%s] updated"%(name))

            if com is not None:
                if len(com) !=3:
                    LOGGER.logger.error("Incorrect dimensions of your com position. 3 elements required")
                    raise
                segment.m_bsp["com"] = com
                LOGGER.logger.info("default com position of the segment [%s] updated"%(name))

            if rog is not None: 
                segment.m_bsp["rog"] = rog
                LOGGER.logger.info("default rog of the segment [%s] updated"%(name))

class Bsp(object):
    """
    the Body Segment Parameters (BSP)  according to Dempster 1995.


    **Reference**

    Dempster. (1955). Body Segment Parameter Data for 2-D Studies

    """
    # TODO: Improve implementation in order to consider different body sement parameter table.


    TABLE = {}
    TABLE["Foot"] = {}
    TABLE["Shank"] = {}
    TABLE["Thigh"] = {}
    TABLE["Thorax"] = {}
    TABLE["UpperArm"] = {}
    TABLE["ForeArm"] = {}
    TABLE["Hand"] = {}
    TABLE["Head"] = {}
    TABLE["Pelvis"] = {}

    TABLE["Pelvis"]["mass"] = 14.2
    TABLE["Pelvis"]["com"] = np.array([ 0.0 ,0.0 , 50.0]) # sagittal - transversal - longitudinal
    TABLE["Pelvis"]["inertia"] = np.array([ 69.0 ,69.0, 0]) # sagittal - transversal - longitudinal

    TABLE["Foot"]["mass"] = 1.45
    TABLE["Foot"]["com"] = np.array([ 0.0 ,0.0 , 50.0]) # sagittal - transversal - longitudinal
    TABLE["Foot"]["inertia"] = np.array([ 69.0 ,69.0, 0]) # sagittal - transversal - longitudinal

    TABLE["Shank"]["mass"] = 4.65
    TABLE["Shank"]["com"] = np.array([ 0 ,0, 43.3]) # sagittal - transversal - longitudinal
    TABLE["Shank"]["inertia"] = np.array([ 52.8 ,52.8, 0]) # sagittal - transversal - longitudinal

    TABLE["Thigh"]["mass"] = 10.0
    TABLE["Thigh"]["com"] = np.array([ 0 ,0, 43.3]) # sagittal - transversal - longitudinal
    TABLE["Thigh"]["inertia"] = np.array([ 54.0 ,54.0, 0]) # sagittal - transversal - longitudinal

    TABLE["Thorax"]["mass"] = 35.5
    TABLE["Thorax"]["com"] = np.array([ 0 ,0, 63.0]) # sagittal - transversal - longitudinal
    TABLE["Thorax"]["inertia"] = np.array([ 0.0 ,0.0, 0.0]) # sagittal - transversal - longitudinal

    TABLE["UpperArm"]["mass"] = 2.8
    TABLE["UpperArm"]["com"] = np.array([ 0 ,0, 43.6]) # sagittal - transversal - longitudinal
    TABLE["UpperArm"]["inertia"] = np.array([ 54.2 ,54.2, 0.0]) # sagittal - transversal - longitudinal

    TABLE["ForeArm"]["mass"] = 1.6
    TABLE["ForeArm"]["com"] = np.array([ 0 ,0, 43.0]) # sagittal - transversal - longitudinal
    TABLE["ForeArm"]["inertia"] = np.array([ 52.6 ,52.6, 0.0]) # sagittal - transversal - longitudinal

    TABLE["Hand"]["mass"] = 0.6
    TABLE["Hand"]["com"] = np.array([ 0 ,0, (( 0.506 )*0.75)*100.0]) # sagittal - transversal - longitudinal 0.75:Knuckle_II_Proportion
    TABLE["Hand"]["inertia"] = np.array([  (0.587*0.75)*100.0 ,(0.587*0.75)*100.0, 0.0]) # sagittal - transversal - longitudinal

    TABLE["Head"]["mass"] = 8.1
    TABLE["Head"]["com"] = np.array([ 0 ,0, (( 0.506 )*0.75)*100.0]) # sagittal - transversal - longitudinal 0.75:Knuckle_II_Proportion
    TABLE["Head"]["inertia"] = np.array([  (0.587*0.75)*100.0 ,(0.587*0.75)*100.0, 0.0]) # sagittal - transversal - longitudinal

    @classmethod
    def setParameters(cls, bspSegmentLabel: str, segmentLength: float, bodymass: float) -> tuple:
        """
        Compute body parameters for a selected lower limb segment.

        Args:
            bspSegmentLabel (str): Segment label defined in the class object `TABLE`.
            segmentLength (float): Length of the segment.
            bodymass (float): Mass of the subject.
    
        Returns:
            tuple: Calculated mass, center of mass, and inertia tensor components for the segment.

        """
        # TODO Pelvis
        # % Length = distance from midpoint of hip joint centres to junction between L4 and L5. (see Winter/Dempster)
        #% 0.925 is found from the ratio of the distance between HJC's and "Length" measured on the current mesh (EthelredBones.OBJ)
        #      PelvisLength = obj.PelvisScale() * 0.925;
        #
        #      CentreOfMass = ( obj.PelvisOriginOffset() + ...
        #        (obj.m_TopLumbar5-obj.PelvisOriginOffset()) * 0.895 ) ./ PelvisLength;
        #
        #      I = Bodymass * 0.142 * ( obj.m_Settings.m_PelvisROG.^2 );
        #      obj.m_KineticPelvis = KineticSegment( obj.m_Pelvis, CentreOfMass, Bodymass*0.142, ...
        #                                            [I;I;I], NullSegment(), [0;0;0] );

        mass = bodymass *  Bsp.TABLE[bspSegmentLabel]["mass"]/100.0
        com = -1.0 * segmentLength *  Bsp.TABLE[bspSegmentLabel]["com"]/100.0 # com from Prox->dist but longitudinal is from Dist-> prox Generally
        ml2 = mass * segmentLength*segmentLength

        Ixx = ml2 * Bsp.TABLE[bspSegmentLabel]["inertia"][0] * Bsp.TABLE[bspSegmentLabel]["inertia"][0] / 10000.0;  # 10000 ( because mm*mm /100, 100 acociount for )
        Iyy = ml2 * Bsp.TABLE[bspSegmentLabel]["inertia"][1] * Bsp.TABLE[bspSegmentLabel]["inertia"][1] / 10000.0;
        Izz = ml2 * Bsp.TABLE[bspSegmentLabel]["inertia"][2] * Bsp.TABLE[bspSegmentLabel]["inertia"][2] / 10000.0;


        return (mass,com,Ixx,Iyy,Izz )

    def __init__(self,iModel:Model):
        self.m_model = iModel

    def compute(self):
        """
        Compute body segment parameters for the model based on Dempster's 1955 body segment parameter data.
        """

        bodymass =  self.m_model.mp["Bodymass"]

#        # example for one segment left thigh
#        length = self.m_model.getSegment("Left Thigh").m_bsp["length"]
#
#        (mass,com,Ixx,Iyy,Izz)  = Bsp.setParameters( "Thigh",length, bodymass)
#        self.m_model.getSegment("Left Thigh").setMass( mass)
#        self.m_model.getSegment("Left Thigh").setComPosition (com)
#        self.m_model.getSegment("Left Thigh").setInertiaTensor (np.array([[Ixx,0.0,0.0],[0.0,Iyy,0.0],[0.0,0.0,Izz]]))


        # automatic method : check if segment Name is in keys of Bsp Table
        for itSegment in self.m_model.m_segmentCollection:
            nameDecompose = itSegment.name.split()

            for it in nameDecompose: # split label along space
                if it in Bsp.TABLE.keys():
                    length = self.m_model.getSegment(itSegment.name).m_bsp["length"]
                    (mass,com,Ixx,Iyy,Izz)  = Bsp.setParameters( it, length, bodymass)
                    if self.m_model.getSegment(itSegment.name).anatomicalFrame.static.getNode_byLabel("com"): # update com if defined during calibration.
                        LOGGER.logger.debug("segment %s -- com already defined during calibration. " %(itSegment.name) )
                        com = self.m_model.getSegment(itSegment.name).anatomicalFrame.static.getNode_byLabel("com").m_local

                    self.m_model.getSegment(itSegment.name).setMass( mass)
                    self.m_model.getSegment(itSegment.name).setComPosition (com)
                    self.m_model.getSegment(itSegment.name).setInertiaTensor (np.array([[Ixx,0.0,0.0],[0.0,Iyy,0.0],[0.0,0.0,Izz]]))
