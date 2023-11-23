# -*- coding: utf-8 -*-
""" this module gathers two classes (`Node` and `Frame`),
important for the construction of a model

A `Frame` is a representation of a coordinate system at a specific time.
Its main attributes are the rotation matrix and the translation vectors.

A `Frame` can collect `Node` instances.
A `Node` represents a 3d point expressed in a local coordinate system

"""
import sys
import math
import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER

# convenient pose functions

def angleAxis_TO_rotationMatrix(anglesAxis):

    """Convert angleaxis  to a 3x3 rotation matrix 

    Args:
        anglesAxis (np.array(3)): global axis at a specific frame

    Returns:
        np.array(3,3): rotation matrix
    """    

    rot = [[0] * 3 for x in range(3)]
    fi = math.sqrt(sum(map(lambda n: n ** 2, anglesAxis)))
    if fi < sys.float_info.epsilon * 100:
        for i in range(3):
            rot[i][i] = 1
        return np.array(rot)

    x = anglesAxis[0] / fi
    y = anglesAxis[1] / fi
    z = anglesAxis[2] / fi

    rot[0][0] = math.cos(fi) + x ** 2 * (1 - math.cos(fi))
    rot[0][1] = x * y * (1 - math.cos(fi)) - z * math.sin(fi)
    rot[0][2] = x * z * (1 - math.cos(fi)) + y * math.sin(fi)

    rot[1][0] = y * x * (1 - math.cos(fi)) + z * math.sin(fi)
    rot[1][1] = math.cos(fi) + y ** 2 * (1 - math.cos(fi))
    rot[1][2] = y * z * (1 - math.cos(fi)) - x * math.sin(fi)

    rot[2][0] = z * x * (1 - math.cos(fi)) - y * math.sin(fi)
    rot[2][1] = z * y * (1 - math.cos(fi)) + x * math.sin(fi)
    rot[2][2] = math.cos(fi) + z ** 2 * (1 - math.cos(fi))


    return np.array(rot)



def rotationMatrix_TO_quaternion(RotMat):
    
    """
    Calculates the quaternion representation from a rotation matrix.

    Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes article "Quaternion Calculus and Fast Animation".

   Args:
      RotMat(array(3,3)):  Rotation matrix

    """

    quaternion = np.zeros((4))
    Trace = np.trace(RotMat)
    if Trace > 0:
        Root = np.sqrt(Trace + 1)
        quaternion[3] = 0.5 * Root
        Root = 0.5 / Root
        quaternion[0] = (RotMat[2, 1] - RotMat[1, 2]) * Root
        quaternion[1] = (RotMat[0, 2] - RotMat[2, 0]) * Root
        quaternion[2] = (RotMat[1, 0] - RotMat[0, 1]) * Root
    else:
        Next = np.array([1, 2, 0])
        i = 0
        if RotMat[1, 1] > RotMat[0, 0]:
            i = 1
        if RotMat[2, 2] > RotMat[i, i]:
            i = 2

        j = Next[i]
        k = Next[j]

        Root = np.sqrt(RotMat[i, i] - RotMat[j, j] - RotMat[k, k] + 1)
        quaternion[i] = 0.5 * Root
        Root = 0.5 / Root
        quaternion[3] = (RotMat[k, j] - RotMat[j, k]) * Root
        quaternion[j] = (RotMat[j, i] + RotMat[i, j]) * Root
        quaternion[k] = (RotMat[k, i] + RotMat[i, k]) * Root

    quaternion = quaternion / np.linalg.norm(quaternion)

    return quaternion


def quaternion_TO_angleAxis(quaternion):

    """
    Calculates the AngleAxis representation from a quaternion


   Args:
       quaternion(array(4)): a quaternion



    """
    if quaternion.ndim != 1: 
        raise Exception("[pyCGM2] - wrong input array dimensions, only 1d")

    imag = quaternion[:-1]
    real = quaternion[3]

    lenQ = np.linalg.norm(imag)
    if lenQ < 100*np.spacing(np.single(1)):
        AngleAxis = imag
    else:
        angle = 2*np.arctan2(lenQ, real)
        AngleAxis = angle/lenQ * imag

    return AngleAxis


    # """Calculates the AngleAxis representation of the rotation described by a
    # quaternion (x,y,z,w)"""
    # if any(map(lambda x: math.isnan(x), quaternion)):
    #     return [float('nan')] * 3

    # imag = quaternion[0:3]
    # real = quaternion[3]

    # length = math.sqrt(sum(map(lambda x: x ** 2, imag)))
    # if length < sys.float_info.epsilon * 100:
    #     AngleAxis = imag
    # else:
    #     angle = 2 * math.atan2(length, real)
    #     AngleAxis = list(map(lambda x: angle / length * x, imag))

    # return AngleAxis

def rotationMatrix_TO_angleAxis(RotMat):


    quaternion = rotationMatrix_TO_quaternion(RotMat)
    AngleAxis = quaternion_TO_angleAxis(quaternion)
    return AngleAxis


def angleAxis_TO_quaternion(anglesAxis):

    RotMat = angleAxis_TO_rotationMatrix(anglesAxis)
    quaternion = rotationMatrix_TO_quaternion(RotMat)

    return quaternion


def quaternion_TO_rotationMatrix(quaternion):
    
    angleAxis = quaternion_TO_angleAxis(quaternion)
    rotMat = angleAxis_TO_rotationMatrix(angleAxis)

    return rotMat


def setFrameData(a1, a2, sequence):
    """
    Set axes and rotation matrix of a coordinate system from 2 vectors and a sequence

    The sequence can be all combinaison of `XYZ`.
    If the second letter is prefixed by `i`, the opposite vector to the cross product a1, a2 is considered

    Args:
       a1(array(1,3)):  first vector
       a2(array(1,3)):  second vector
       sequence(str):  construction sequence (XYZ, XYiZ,...)

    """

    if sequence == "XYZ" or sequence == "XYiZ":
        if sequence == "XYiZ":
            a2 = a2*-1.0
        axisX = a1
        axisY = a2
        axisZ = np.cross(a1, a2)
        rot = np.array([axisX, axisY, axisZ]).T

    if sequence == "XZY" or sequence == "XZiY":
        if sequence == "XZiY":
            a2 = a2*-1.0
        axisX = a1
        axisZ = a2
        axisY = np.cross(a2, a1)
        rot = np.array([axisX, axisY, axisZ]).T

    if sequence == "YZX" or sequence == "YZiX":
        if sequence == "YZiX":
            a2 = a2*-1.0
        axisY = a1
        axisZ = a2
        axisX = np.cross(a1, a2)
        rot = np.array([axisX, axisY, axisZ]).T

    if sequence == "YXZ" or sequence == "YXiZ":
        if sequence == "YXiZ":
            a2 = a2*-1.0
        axisY = a1
        axisX = a2
        axisZ = np.cross(a2, a1)
        rot = np.array([axisX, axisY, axisZ]).T

    if sequence == "YXZ" or sequence == "YXiZ":
        if sequence == "YXiZ":
            a2 = a2*-1.0
        axisY = a1
        axisX = a2
        axisZ = np.cross(a2, a1)
        rot = np.array([axisX, axisY, axisZ]).T

    if sequence == "ZXY" or sequence == "ZXiY":
        if sequence == "ZXiY":
            a2 = a2*-1.0
        axisZ = a1
        axisX = a2
        axisY = np.cross(a1, a2)
        rot = np.array([axisX, axisY, axisZ]).T

    if sequence == "ZYX" or sequence == "ZYiX":
        if sequence == "ZYiX":
            a2 = a2*-1.0
        axisZ = a1
        axisY = a2
        axisX = np.cross(a2, a1)
        rot = np.array([axisX, axisY, axisZ]).T

    return axisX, axisY, axisZ, rot


class Node(object):
    """
    A node is a local position of a 3D point in a coordinate system

    Note:
        Automatically, the suffix "_node" ends the node label

    Args:
       label(str):  desired label of the node
       desc(str,Optional):  description
    """

    def __init__(self, label, desc=""):
        self.m_name = label+"_node"
        self.m_global = np.zeros((1, 3))
        self.m_local = np.zeros((1, 3))
        self.m_desc = desc

    def computeLocal(self, rot, t):
        """
        Compute local position from global position

        Args:
            rot(array(3,3)):  a rotation matrix
            t(array((1,3))):  a translation vector

        """
        self.m_local = np.dot(rot.T, (self.m_global-t))

    def computeGlobal(self, rot, t):
        """
        Compute global position from local

        Args:
            rot(array((3,3)):  a rotation matrix
            t(array((1,3)):  a translation vector

        """

        self.m_global = np.dot(rot, self.m_local) + t

    def setDescription(self, description):
        """
        Set description of the node

        Args:
            description(str):  description
        """
        self.m_desc = description

    def getLabel(self):
        """
        Get the node label
        """
        label = self.m_name
        return label[0:label.find("_node")]

    def getDescription(self):
        """
        Get the node description
        """

        return self.m_desc

    def getLocal(self):
        """
        Get the local coordinates  of the node
        """

        return self.m_local

    def getGlobal(self):
        """
        Get the global coordinates  of the node
        """
        return self.m_global


class Frame(object):
    """
    A `Frame` represents a coordinate system
    """

    def __init__(self):

        self.m_axisX = np.zeros((1, 3))
        self.m_axisY = np.zeros((1, 3))
        self.m_axisZ = np.zeros((1, 3))

        self._translation = np.zeros((1, 3))
        self._matrixRot = np.zeros((3, 3))
        self._anglesAxis = None
        self._quaternion = None

        self._nodes = []

    def constructFromAnglesAxis(self,angleAxisValues ):
        
        self._anglesAxis = angleAxisValues
        self._matrixRot = angleAxis_TO_rotationMatrix(angleAxisValues)

        self.m_axisX = self._matrixRot[:, 0]
        self.m_axisY = self._matrixRot[:, 1]
        self.m_axisZ = self._matrixRot[:, 2]


    def constructFromQuaternion(self,quaternionValues):

        self._quaternion = quaternionValues
        self._matrixRot = quaternion_TO_rotationMatrix(quaternionValues) 

        
        self.m_axisX = self._matrixRot[:, 0]
        self.m_axisY = self._matrixRot[:, 1]
        self.m_axisZ = self._matrixRot[:, 2]
        
        
    def setAxes(self,x,y,z):

        self.m_axisX = x
        self.m_axisY = y
        self.m_axisZ = z

        self._matrixRot[:, 0] = x
        self._matrixRot[:, 1] = y
        self._matrixRot[:, 2] = z

    def getRotation(self):
        """
        Get rotation matrix

        Returns:
            array((3,3):  a rotation matrix

        """
        return self._matrixRot

    def getAngleAxis(self):
        """
        Get the angle axis
        """
        if self._anglesAxis is not None:
            return self._anglesAxis
        else: 
            axisAngle = rotationMatrix_TO_angleAxis(self._matrixRot)

            return axisAngle

    def getQuaternion(self):
        """
        Get the quaternion
        """
        if self._quaternion is not None:
            return self._quaternion
        else: 
            quaternion = rotationMatrix_TO_quaternion(self._matrixRot)
            return quaternion

    def getTranslation(self):
        """
        Get translation vector
        """
        return self._translation

    def setRotation(self, R):
        """
        Set the rotation matrix

        Args:
           R(array(3,3):  a rotation matrix
        """

        self._matrixRot = R

        self.m_axisX = R[:, 0]
        self.m_axisY = R[:, 1]
        self.m_axisZ = R[:, 2]

        self._quaternion = rotationMatrix_TO_quaternion(self._matrixRot)
        self._anglesAxis = rotationMatrix_TO_angleAxis(self._matrixRot)

    def setTranslation(self, t):
        """
        Set the translation vector

        Args:
           t(array(3,)):  a translation vector
        """
        self._translation = t

    def updateAxisFromRotation(self, R):
        """
        Update the rotation matrix

        Args:
           R(array(3,3):  a rotation matrix
        """
        self.m_axisX = R[:, 0]
        self.m_axisY = R[:, 1]
        self.m_axisZ = R[:, 2]

        self._matrixRot = R

        self._quaternion = rotationMatrix_TO_quaternion(self._matrixRot)
        self._anglesAxis = rotationMatrix_TO_angleAxis(self._matrixRot)

    def update(self, R, t):
        """
        Update both the rotation matrix and the translation vector

        Args:
           R(array(3,3):  a rotation matrix
           t(array(3,)):  a translation vector
        """

        self.m_axisX = R[:, 0]
        self.m_axisY = R[:, 1]
        self.m_axisZ = R[:, 2]
        self._translation = t
        self._matrixRot = R

        self._quaternion = rotationMatrix_TO_quaternion(self._matrixRot)
        self._anglesAxis = rotationMatrix_TO_angleAxis(self._matrixRot)

    def addNode(self, nodeLabel, position, positionType="Global", desc=""):
        """
        Append a `Node` to a Frame

        Args:
            nodeLabel(str):  node label
            position(array(3,)):  a translation vector
            positionType(str,Optional):  two choice Global or Local
            desc(str,Optional):  description

        """
        #TODO : use an Enum for the argment positionType

        LOGGER.logger.debug("new node (%s) added " % nodeLabel)

        isFind = False
        i = 0
        for nodeIt in self._nodes:
            if str(nodeLabel+"_node") == nodeIt.m_name:
                isFind = True
                index = i
            i += 1

        if isFind:
            if positionType == "Global":
                self._nodes[index].m_global = position
                self._nodes[index].computeLocal(
                    self._matrixRot, self._translation)
            elif positionType == "Local":
                self._nodes[index].m_local = position
                self._nodes[index].computeGlobal(
                    self._matrixRot, self._translation)
            else:
                raise Exception("positionType not Known (Global or Local")

            LOGGER.logger.debug(
                "[pyCGM2] node (%s) values updated" % (nodeLabel))
            previousDesc = self._nodes[index].m_desc
            if previousDesc != desc:
                LOGGER.logger.debug(
                    "[pyCGM2] node (%s) description updated [%s -> %s]" % (nodeLabel, previousDesc, desc))
                self._nodes[index].m_desc = desc

        else:
            node = Node(nodeLabel, desc=desc)
            if positionType == "Global":
                node.m_global = position
                node.computeLocal(self._matrixRot, self._translation)
            elif positionType == "Local":
                node.m_local = position
                node.computeGlobal(self._matrixRot, self._translation)
            else:
                raise Exception("positionType not Known (Global or Local")
            self._nodes.append(node)

    def getNode_byIndex(self, index):
        """
        Get a `node` from its index

        Args:
            index(int):  index of the node within the list

        """
        return self._nodes[index]

    def getNode_byLabel(self, label):
        """
        Get a `node` from its label

         Args:
            label(str):  the node label

        """

        for nodeIt in self._nodes:
            if str(label+"_node") == nodeIt.m_name:
                LOGGER.logger.debug(
                    " target label ( %s):  label find (%s) " % (label, nodeIt.m_name))
                return nodeIt
            #else:
            #    raise Exception("Node label (%s) not found " %(label))

        return False

    def getNodeLabels(self, display=True):
        """
        Display all node labels
        """
        labels = []
        for nodeIt in self._nodes:
            labels.append(nodeIt.m_name[:-5])

            if display:
                print(nodeIt.m_name)

        return labels

    def eraseNodes(self):
        """
        Erase all nodes
        """
        self._nodes = []

    def getNodes(self):
        """
        Return all node instances
        """

        return self._nodes

    def isNodeExist(self, nodeLabel):
        """
        Check if a node exists from its label

        Args:
            nodeLabel (str): the node label
        """

        flag = False
        for nodeIt in self._nodes:
            if str(nodeLabel+"_node") == nodeIt.m_name:
                LOGGER.logger.debug(
                    " target label ( %s):  label find (%s) " % (nodeLabel, nodeIt.m_name))
                flag = True
                break

        if not flag:
            LOGGER.logger.debug(
                " node label ( %s) doesn t exist " % (nodeLabel))

        return flag

    def getNodeIndex(self, nodeLabel):
        """
        Get the node index

        Args:
            nodeLabel (str): the node label
        """

        if self.isNodeExist(nodeLabel):
            i = 0
            for nodeIt in self._nodes:
                if str(nodeLabel+"_node") == nodeIt.m_name:
                    index = i
                    break
                i += 1
            return i
        else:
            raise Exception("[pyCGM2] node label doesn t exist")

    def updateNode(self, nodeLabel, localArray, globalArray, desc=""):
        """Update a node

        Args:
            nodeLabel (str): the node label
            localArray (array(3)): local position
            globalArray (array(3)): global position
            desc (str,Optional): description

        """


        index = self.getNodeIndex(nodeLabel)

        self._nodes[index].m_global = globalArray
        self._nodes[index].m_local = localArray
        self._nodes[index].m_desc = desc

    def copyNode(self, nodeLabel, nodeToCopy):
        """
        Copy a node and add it or update a existant node

        Args:
            nodeLabel (str): the node label
            nodeToCopy (str): the label of the node to copy
        """

        indexToCopy = self.getNodeIndex(nodeToCopy)
        globalArray = self._nodes[indexToCopy].m_global
        localArray = self._nodes[indexToCopy].m_local
        desc = self._nodes[indexToCopy].m_desc

        if self.isNodeExist(nodeLabel):
            index = self.getNodeIndex(nodeToCopy)
            self._nodes[index].m_global = globalArray
            self._nodes[index].m_local = localArray
            self._nodes[index].m_desc = desc
        else:
            self.addNode(nodeLabel, globalArray, position="Global", desc=desc)

    def getGlobalPosition(self, nodeLabel):
        """Return the global position

        Args:
            nodeLabel (str): the node label

        """

        node = self.getNode_byLabel(nodeLabel)

        return np.dot(self.getRotation(), node.getLocal()) + self.getTranslation()
