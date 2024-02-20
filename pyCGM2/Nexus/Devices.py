"""
The module is a viconnexus interface. 
It contains Objects (ie Device) contructed from vicon nexus api
"""

import numpy as np

import pyCGM2; LOGGER = pyCGM2.LOGGER

try:
    from viconnexusapi import ViconNexus
except ImportError as e:
    LOGGER.logger.error(f"viconnexusapi not installed: {e}")

from typing import List, Tuple, Dict, Optional,Union

class Channel(object):
    """
    Represents a channel in a pyCGM2-Nexus interface.

    Attributes:
        label (str): The label of the channel.
        values (np.ndarray): The numerical values associated with the channel.
        unit (str): The unit of measurement for the channel's values.
        description (str): A brief description of the channel.
    """

    def __init__(self, label:str,values:np.ndarray,unit:str,description:str):
        """
        Initializes the Channel object with specified label, values, unit, and description.
        """
        self.m_label = label
        self.m_values = values
        self.m_description = description
        self.m_unit = unit

    def getLabel(self):
        """
        Returns the label of the channel.

        Returns:
            str: The label of the channel.
        """
        return  self.m_label
    def getValues(self):
        """
        Returns the values of the channel.

        Returns:
            np.ndarray: The numerical values of the channel.
        """
        return self.m_values
    def getDescription(self):
        """
        Returns the description of the channel.

        Returns:
            str: A brief description of the channel.
        """
        return  self.m_description
    def getUnit(self):
        """
        Returns the unit of measurement for the channel's values.

        Returns:
            str: The unit of measurement of the channel.
        """
        return self.m_unit



class Device(object):
    """
    Represents a generic device in a pyCGM2-Nexus interface.

    Attributes:
        NEXUS (ViconNexus.ViconNexus): The Vicon Nexus handle.
        id (str): The identifier of the device.
    """

    def __init__(self,NEXUS, id:str):
        """Initializes the Device object with a Vicon Nexus handle and device ID.
        """
        self.NEXUS = NEXUS
        self.m_id = id

        self.m_name = self.NEXUS.GetDeviceDetails(self.m_id)[0]
        self.m_type = self.NEXUS.GetDeviceDetails(self.m_id)[1]
        self.m_frequency = self.NEXUS.GetDeviceDetails(self.m_id)[2]
        self.m_outputIds = self.NEXUS.GetDeviceDetails(self.m_id)[3]

        self.m_forcePlateInfo = self.NEXUS.GetDeviceDetails(self.m_id)[4]
        self.m_eyeTrackerInfo = self.NEXUS.GetDeviceDetails(self.m_id)[5]


    def getDeviceName(self):
        """
        Returns the name of the device.

        Returns:
            str: The name of the device.
        """
        return self.m_name

    def getDeviceFrequency(self):
        """
        Returns the sampling frequency of the device.

        Returns:
            float: The sampling frequency of the device in Hertz.
        """
        return self.m_frequency

    def getOutputNames(self):
        """
        Returns a list of output names from the device.

        Returns:
            List[str]: A list of output names associated with the device.
        """
        out = []
        for i in self.m_outputIds:
            out.append(self.NEXUS.GetDeviceOutputDetails(self.m_id,i)[0])
        return out



class AnalogDevice(Device):
    """
    Represents an analog device in a pyCGM2-Nexus interface, extending the Device class.

    Attributes:
        channels (list[Channel]): A list of channels associated with the analog device.
    """

    def __init__(self,NEXUS, id:str):
        """Initializes the AnalogDevice object with a Vicon Nexus handle and device ID.
        """
        super(AnalogDevice, self).__init__(NEXUS,id)
        self.m_channels = []
        #self.m_id = id

    def getUnit(self):
        """
        Returns the unit of measurement for the analog device's channels.

        Returns:
            str: The unit of measurement for the channels.
        """
        unit = []
        for outputId in self.m_outputIds:
            unit.append(str(self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[2]))
        return  unit


    def getChannels(self):
        """
        Retrieves and stores the channels associated with the analog device.

        Returns:
            List[Channel]: A list of Channel objects associated with the device.
        """
        for outputId in self.m_outputIds:
            outputName = self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]
            outputType = self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[1]
            unit = str(self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[2])
            i=0
            for channelId in self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[5]:
                label = str(outputName) +"."+ str(self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4][i])
                values = np.asarray(self.NEXUS.GetDeviceChannel(self.m_id,outputId,channelId)[0])
                description =  "Analog::"+str(outputType)+" ["+str(self.m_id) + "," + str(channelId)+"]"
                self.m_channels.append(Channel(label,values,unit,description))
                i+=1

        return self.m_channels



class ForcePlate(Device):
    """
    Represents a force plate device in a pyCGM2-Nexus interface, extending the Device class.

    Attributes:
        forcePlateInfo: Information about the force plate, such as origin and orientation.
    """

    def __init__(self,NEXUS, id:str):
        """Initializes the ForcePlate object with a Vicon Nexus handle and device ID.
        """
        super(ForcePlate, self).__init__(NEXUS,id)


    def getDescription(self):
        """
        Returns a description of the force plate.

        Returns:
            str: A description of the force plate.
        """
        return str("Force Plate [" +str(self.m_id) + "]")
#
    def getOrigin(self):
        """
        Returns the origin of the force plate in global coordinates.

        Returns:
            np.ndarray: The origin coordinates of the force plate.
        """
        nfp_info = self.m_forcePlateInfo
        return np.asarray(nfp_info.WorldT)

    def getLocalOrigin(self):
        """
        Returns the local origin of the force plate.

        Returns:
            np.ndarray: The local origin coordinates of the force plate.
        """
        nfp_info = self.m_forcePlateInfo
        return np.asarray(nfp_info.LocalT)


    def getContext(self):
        """
        Returns the context of the force plate, such as the side it represents.

        Returns:
            str: The context of the force plate.
        """
        nfp_info = self.m_forcePlateInfo
        return nfp_info.Context

    def getPhysicalOrigin(self):
        """
        Returns the physical origin location of the force plate.

        Returns:
            np.ndarray: The physical origin coordinates of the force plate.
        """
        nfp_info = self.m_forcePlateInfo
        worldR =  np.asarray(nfp_info.WorldR).reshape((3,3))
        origin = self.getOrigin()
        return origin + np.dot(worldR, np.asarray(nfp_info.LocalT))



    def getOrientation(self):
        """
        Returns the orientation of the force plate.

        Returns:
            np.ndarray: The orientation matrix of the force plate.
        """
        nfp_info = self.m_forcePlateInfo
        worldR =  np.asarray(nfp_info.WorldR).reshape((3,3))
        localR =  np.asarray(nfp_info.LocalR).reshape((3,3))

        return np.dot(worldR,localR)

    def getForceUnit(self):
        """
        Returns the unit of measurement for the force data from the force plate.

        Returns:
            str: The unit of force measurement.
        """

        for outputId in self.m_outputIds:
            if self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Fx","Fy","Fz"]:
                outputName =  self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = self.NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)
        return str(self.NEXUS.GetDeviceOutputDetails(self.m_id,deviceOutputID)[2])

    def getMomentUnit(self):
        """
        Returns the unit of measurement for the moment data from the force plate.

        Returns:
            str: The unit of moment measurement.
        """

        for outputId in self.m_outputIds:
            if self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Mx","My","Fz"]:
                outputName =  self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]


        deviceOutputID = self.NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)
        return str(self.NEXUS.GetDeviceOutputDetails(self.m_id,deviceOutputID)[2])



    def getGlobalForce(self):
        """
        Returns the force data from the force plate in global coordinates.

        Returns:
            np.ndarray: The global force data from the force plate.
        """

        for outputId in self.m_outputIds:
            if self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Fx","Fy","Fz"]:
                outputName =  self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = self.NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)

        channelID_x = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Fx')
        channelID_y = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Fy')
        channelID_z = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Fz')

        x = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_x )[0])
        y = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_y )[0])
        z = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_z )[0])

        return np.array([x,y,z]).T


    def getGlobalMoment(self):
        """
        Returns the moment data from the force plate in global coordinates.

        Returns:
            np.ndarray: The global moment data from the force plate.
        """
        for outputId in self.m_outputIds:
            if self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Mx","My","Mz"]:
                outputName =  self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = self.NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)

        channelID_x = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Mx')
        channelID_y = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'My')
        channelID_z = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Mz')

        x = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_x )[0])
        y = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_y )[0])
        z = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_z )[0])

        return np.array([x,y,z]).T


    def getGlobalCoP(self):
        """
        Returns the Center of Pressure (CoP) from the force plate in global coordinates.

        Returns:
            np.ndarray: The global CoP data from the force plate.
        """
        for outputId in self.m_outputIds:
            if self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Cx","Cy","Cz"]:
                outputName =  self.NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = self.NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)

        channelID_x = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Cx')
        channelID_y = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Cy')
        channelID_z = self.NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Cz')

        x = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_x )[0])
        y = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_y )[0])
        z = np.asarray(self.NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_z )[0])

        return np.array([x,y,z]).T


    def getLocalReactionForce(self):
        """
        Returns the reaction force from the force plate in local coordinates.

        Returns:
            np.ndarray: The local reaction force data from the force plate.
        """


        force = -1* self.getGlobalForce()
        R = self.getOrientation()

        out = np.zeros(force.shape)
        for i in range(0, force.shape[0]):
            out[i,:] = np.dot(R.T,force[i,:])
        return out

    def getLocalReactionMoment(self):
        """
        Returns the reaction moment from the force plate in local coordinates.

        Returns:
            np.ndarray: The local reaction moment data from the force plate.
        """
        moment = self.getGlobalMoment()
        force = self.getGlobalForce()
        #origin = -1*self.getLocalPhysicOrigin()

        origin = self.getOrigin() - self.getPhysicalOrigin()

        moment_transp = np.zeros(moment.shape)
        moment_transp[:,0] =  moment[:,0] - (force[:,1] *  origin[2] - origin[1]*force[:,2])
        moment_transp[:,1] =  moment[:,1] - (force[:,2] *  origin[0] - origin[2]*force[:,0])
        moment_transp[:,2] =  moment[:,2] - (force[:,0] *  origin[1] - origin[0]*force[:,1])

        moment = -1* moment_transp#self.getGlobalMoment()
        R = self.getOrientation()



        out = np.zeros(moment.shape)
        for i in range(0, moment.shape[0]):
            out[i,:] = np.dot(R.T,moment[i,:])
        return out



    def getCorners(self):
        """
        Returns the coordinates of the corners of the force plate.

        Returns:
            np.ndarray: The coordinates of the corners of the force plate.
        """
        nfp_info = self.m_forcePlateInfo

        R = np.asarray(nfp_info.WorldR).reshape((3,3))
        t = np.asarray(nfp_info.WorldT)

        corner1 = np.asarray(nfp_info.UpperBounds)
        corner3 = np.asarray(nfp_info.LowerBounds)
        corner2 = np.asarray(nfp_info.UpperBounds)*np.array([1,-1,0])
        corner0 = np.asarray(nfp_info.LowerBounds)*np.array([1,-1,0])

        corners = np.zeros((3,4))
        corners[:,0] = np.dot(R,corner0)+t #2
        corners[:,1] = np.dot(R,corner1)+t
        corners[:,2] = np.dot(R,corner2)+t
        corners[:,3] = np.dot(R,corner3)+t

        return corners
