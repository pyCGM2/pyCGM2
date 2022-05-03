# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Nexus
#APIDOC["Draft"]=False
#--end--

"""
The module is a viconnexus interface. it contains Object (ie Device) which can be contructed from vicon nexus api
"""

from viconnexusapi import ViconNexus
import numpy as np

import pyCGM2; LOGGER = pyCGM2.LOGGER

try:
    import ViconNexus
except:
    from viconnexusapi import ViconNexus

try:
    NEXUS = ViconNexus.ViconNexus()
except:
    LOGGER.logger.error("Nexus is not running")

class Channel(object):
    """Channel

    Args:
        label (str): channel label
        values (array): values
        unit (str): init
        description (str): short description of the channel
    """

    def __init__(self, label,values,unit,description):
        self.m_label = label
        self.m_values = values
        self.m_description = description
        self.m_unit = unit

    def getLabel(self):
        """ return the channel label"""
        return  self.m_label
    def getValues(self):
        """ return the channel values"""
        return self.m_values
    def getDescription(self):
        """ return the channel description"""
        return  self.m_description
    def getUnit(self):
        """ return the channel unit"""
        return self.m_unit



class Device(object):
    """Device

    Args:
        id (str): device ID

    """

    def __init__(self, id):
        self.m_id = id

        self.m_name = NEXUS.GetDeviceDetails(self.m_id)[0]
        self.m_type = NEXUS.GetDeviceDetails(self.m_id)[1]
        self.m_frequency = NEXUS.GetDeviceDetails(self.m_id)[2]
        self.m_outputIds = NEXUS.GetDeviceDetails(self.m_id)[3]

        self.m_forcePlateInfo = NEXUS.GetDeviceDetails(self.m_id)[4]
        self.m_eyeTrackerInfo = NEXUS.GetDeviceDetails(self.m_id)[5]


    def getDeviceName(self):
        """return device name"""
        return self.m_name

    def getDeviceFrequency(self):
        """return device sample m_frequency"""
        return self.m_frequency

    def getOutputNames(self):
        """return the list of ouputs"""
        out = list()
        for i in self.m_outputIds:
            out.append(NEXUS.GetDeviceOutputDetails(self.m_id,i)[0])
        return out



class AnalogDevice(Device):
    """Analog Device

    Args:
        id (str): device ID

    """

    def __init__(self, id):
        super(AnalogDevice, self).__init__(id)
        self.m_channels = list()
        #self.m_id = id

    def getUnit(self):
        """ return device unit"""
        unit = list()
        for outputId in self.m_outputIds:
            unit.append(str(NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[2]))
        return  unit


    def getChannels(self):
        """ return channel names"""

        for outputId in self.m_outputIds:
            outputName = NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]
            outputType = NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[1]
            unit = str(NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[2])
            i=0
            for channelId in NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[5]:
                label = str(outputName) +"."+ str(NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4][i])
                values = np.asarray(NEXUS.GetDeviceChannel(self.m_id,outputId,channelId)[0])
                description =  "Analog::"+str(outputType)+" ["+str(self.m_id) + "," + str(channelId)+"]"
                self.m_channels.append(Channel(label,values,unit,description))
                i+=1

        return self.m_channels



class ForcePlate(Device):
    """force plate device

    Args:
        id (str): device ID

    """

    def __init__(self, id):
        super(ForcePlate, self).__init__(id)


    def getDescription(self):
        """ return force plate description """
        return str("Force Plate [" +str(self.m_id) + "]")
#
    def getOrigin(self):
        """ return force plate origin """
        nfp_info = self.m_forcePlateInfo
        return np.asarray(nfp_info.WorldT)

    def getLocalOrigin(self):
        """ return force plate local Origin """
        nfp_info = self.m_forcePlateInfo
        return np.asarray(nfp_info.LocalT)


    def getContext(self):
        """ return event context """
        nfp_info = self.m_forcePlateInfo
        return nfp_info.Context

    def getPhysicalOrigin(self):
        """ return physical origin location """
        nfp_info = self.m_forcePlateInfo
        worldR =  np.asarray(nfp_info.WorldR).reshape((3,3))
        origin = self.getOrigin()
        return origin + np.dot(worldR, np.asarray(nfp_info.LocalT))



    def getOrientation(self):
        """ return force plate orientation """
        nfp_info = self.m_forcePlateInfo
        worldR =  np.asarray(nfp_info.WorldR).reshape((3,3))
        localR =  np.asarray(nfp_info.LocalR).reshape((3,3))

        return np.dot(worldR,localR)

    def getForceUnit(self):
        """ return force unit """

        for outputId in self.m_outputIds:
            if NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Fx","Fy","Fz"]:
                outputName =  NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)
        return str(NEXUS.GetDeviceOutputDetails(self.m_id,deviceOutputID)[2])

    def getMomentUnit(self):
        """ return moment unit """

        for outputId in self.m_outputIds:
            if NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Mx","My","Fz"]:
                outputName =  NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]


        deviceOutputID = NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)
        return str(NEXUS.GetDeviceOutputDetails(self.m_id,deviceOutputID)[2])



    def getGlobalForce(self):
        """ return force in the global coordinate system """

        for outputId in self.m_outputIds:
            if NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Fx","Fy","Fz"]:
                outputName =  NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)

        channelID_x = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Fx')
        channelID_y = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Fy')
        channelID_z = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Fz')

        x = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_x )[0])
        y = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_y )[0])
        z = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_z )[0])

        return np.array([x,y,z]).T


    def getGlobalMoment(self):
        """ return moment in the global coordinate system """

        for outputId in self.m_outputIds:
            if NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Mx","My","Mz"]:
                outputName =  NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)

        channelID_x = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Mx')
        channelID_y = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'My')
        channelID_z = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Mz')

        x = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_x )[0])
        y = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_y )[0])
        z = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_z )[0])

        return np.array([x,y,z]).T


    def getGlobalCoP(self):
        """ return COP in the global coordinate system """

        for outputId in self.m_outputIds:
            if NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[4] == ["Cx","Cy","Cz"]:
                outputName =  NEXUS.GetDeviceOutputDetails(self.m_id,outputId)[0]

        deviceOutputID = NEXUS.GetDeviceOutputIDFromName(self.m_id,outputName)

        channelID_x = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Cx')
        channelID_y = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Cy')
        channelID_z = NEXUS.GetDeviceChannelIDFromName(self.m_id, deviceOutputID, 'Cz')

        x = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_x )[0])
        y = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_y )[0])
        z = np.asarray(NEXUS.GetDeviceChannelGlobal( self.m_id, deviceOutputID, channelID_z )[0])

        return np.array([x,y,z]).T


    def getLocalReactionForce(self):
        """ return force in the local force plate coordinate system """

        force = -1* self.getGlobalForce()
        R = self.getOrientation()

        out = np.zeros(force.shape)
        for i in range(0, force.shape[0]):
            out[i,:] = np.dot(R.T,force[i,:])
        return out

    def getLocalReactionMoment(self):
        """ return moment in the local force plate coordinate system """

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
        """ return corners location in the coordinate system """
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
