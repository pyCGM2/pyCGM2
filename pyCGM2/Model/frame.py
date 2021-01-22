# -*- coding: utf-8 -*-
import numpy as np
import logging



def getQuaternionFromMatrix(RotMat):
    """
        Calculates the quaternion representation of the rotation described by RotMat
        Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes article "Quaternion Calculus and Fast Animation".

       :Parameters:
           - `RotMat` (numy.array(3,3)) - Rotation Matrix


        :Return:
            - `Quaternion` (numy.array(4)) - 4 components of a quaternion

    """

    Quaternion = np.zeros( (4) );
    Trace = np.trace( RotMat )
    if Trace > 0:
        Root = np.sqrt( Trace + 1 )
        Quaternion[3] = 0.5 * Root
        Root = 0.5 / Root;
        Quaternion[0] = ( RotMat[2,1] - RotMat[1,2] ) * Root
        Quaternion[1] = ( RotMat[0,2] - RotMat[2,0] ) * Root
        Quaternion[2] = ( RotMat[1,0] - RotMat[0,1] ) * Root
    else:
        Next = np.array([ 1, 2, 0 ])
        i = 0
        if RotMat[1,1] > RotMat[0,0]:
            i = 1
        if RotMat[2,2] > RotMat[i,i] :
            i = 2

        j = Next[i]
        k = Next[j]

        Root = np.sqrt( RotMat[i,i] - RotMat[j,j] - RotMat[k,k] + 1 )
        Quaternion[i] = 0.5 * Root
        Root = 0.5 / Root;
        Quaternion[3] = ( RotMat[k,j] - RotMat[j,k] ) * Root
        Quaternion[j] = ( RotMat[j,i] + RotMat[i,j] ) * Root
        Quaternion[k] = ( RotMat[k,i] + RotMat[i,k] ) * Root

    Quaternion = Quaternion / np.linalg.norm( Quaternion)

    return Quaternion


def angleAxisFromQuaternion(Quaternion):
    """
        Calculates the AngleAxis representation of the rotation described by a
        quaternion

       :Parameters:
           - `Quaternion` (numy.array(4)) - 4 components of a quaternion


        :Return:
            - `AngleAxis` (numy.array(3)) - angle Axis in deg

    """

    imag = Quaternion[:-1 ]
    real = Quaternion[ 3 ]

    lenQ = np.linalg.norm( imag )
    if lenQ < 100*np.spacing(np.single(1)):
        AngleAxis = imag
    else:
        angle = 2*np.arctan2( lenQ, real )
        AngleAxis = angle/lenQ * imag

    return np.rad2deg(AngleAxis)

def setFrameData(a1,a2,sequence):
    """
        set Frame of a ccordinate system accoring two vector and a sequence

        :Parameters:
           - `a1` (numy.array(1,3)) - first vector
           - `a2` (str) - second vector
           - `sequence` (str) - construction sequence (XYZ, XYiZ)

        :Return:
            - `axisX` (numy.array(1,3)) - x-axis of the coordinate system
            - `axisY` (numy.array(1,3)) - y-axis of the coordinate system
            - `axisZ` (numy.array(1,3)) - z-axis of the coordinate system
            - `rot` (numy.array(3,3)) - rotation matrix of the coordinate system

        .. note:: if sequence includes a *i* ( ex: XYiZ), opposite of vector a2 is considered

    """

    if sequence == "XYZ" or sequence == "XYiZ" :
        if sequence == "XYiZ":
            a2=a2*-1.0
        axisX=a1
        axisY=a2
        axisZ=np.cross(a1,a2)
        rot=np.array([axisX,axisY,axisZ]).T

    if sequence == "XZY" or sequence == "XZiY" :
        if sequence == "XZiY":
            a2=a2*-1.0
        axisX=a1
        axisZ=a2
        axisY=np.cross(a2,a1)
        rot=np.array([axisX,axisY,axisZ]).T

    if sequence == "YZX" or sequence == "YZiX" :
        if sequence == "YZiX":
            a2=a2*-1.0
        axisY=a1
        axisZ=a2
        axisX=np.cross(a1,a2)
        rot=np.array([axisX,axisY,axisZ]).T

    if sequence == "YXZ" or sequence == "YXiZ" :
        if sequence == "YXiZ":
            a2=a2*-1.0
        axisY=a1
        axisX=a2
        axisZ=np.cross(a2,a1)
        rot=np.array([axisX,axisY,axisZ]).T


    if sequence == "YXZ" or sequence == "YXiZ" :
        if sequence == "YXiZ":
            a2=a2*-1.0
        axisY=a1
        axisX=a2
        axisZ=np.cross(a2,a1)
        rot=np.array([axisX,axisY,axisZ]).T


    if sequence == "ZXY" or sequence == "ZXiY" :
        if sequence == "ZXiY":
            a2=a2*-1.0
        axisZ=a1
        axisX=a2
        axisY=np.cross(a1,a2)
        rot=np.array([axisX,axisY,axisZ]).T

    if sequence == "ZYX" or sequence == "ZYiX" :
        if sequence == "ZYiX":
            a2=a2*-1.0
        axisZ=a1
        axisY=a2
        axisX=np.cross(a2,a1)
        rot=np.array([axisX,axisY,axisZ]).T

    return axisX, axisY, axisZ, rot

class Node(object):
    """
        A node is a local position of a point in a Frame
    """

    def __init__(self,label,desc = ""):
        """
            :Parameters:
               - `label` (str) - desired label of the node

            .. note:: automatically, the suffix "_node" ends the node label
        """

        self.m_name = label+"_node"
        self.m_global = np.zeros((1,3))
        self.m_local = np.zeros((1,3))
        self.m_desc = desc

    def computeLocal(self,rot,t):
        """
            Compute local position from global position

            :Parameters:
                - `rot` (np.array(3,3)) - a rotation matrix
                - `t` (np.array((1,3))) - a translation vector

        """
        self.m_local=np.dot(rot.T,(self.m_global-t))

    def computeGlobal(self,rot,t):
        """
            Compute global position from local

            :Parameters:
                - `rot` (np.array((3,3))) - a rotation matrix
                - `t` (np.array((1,3))) - a translation vector

        """

        self.m_global=np.dot(rot,self.m_local) +t

    def setDescription(self, description):
        self.m_desc = description

    def getLabel(self):
        label = self.m_name
        return label[0:label.find("_node")]


    def getDescription(self):

        return self.m_desc


    def getLocal(self):

        return self.m_local

    def getGlobal(self):

        return self.m_global



class Frame(object):
    """
        A Frame defined a segment pose

    """




    def __init__(self):

        self.m_axisX=np.zeros((1,3))
        self.m_axisY=np.zeros((1,3))
        self.m_axisZ=np.zeros((1,3))

        self._translation=np.zeros((1,3))
        self._matrixRot=np.zeros((3,3))


        self._nodes=[]

    def getRotation(self):
        """
            Get rotation matrix

            :Return:
                - `na` (np.array((3,3))) - a rotation matrix

        """
        return self._matrixRot

    def getAngleAxis(self):

        quaternion = getQuaternionFromMatrix(self._matrixRot)
        axisAngle =  angleAxisFromQuaternion(quaternion)


        return axisAngle


    def getTranslation(self):
        """
            Get translation vector

            :Return:
                - `na` (np.array((3,))) - a translation vector

        """
        return self._translation

    def setRotation(self, R):
        """
            Set rotation matrix

            :Parameters:
               - `R` (np.array(3,3) - a rotation matrix
        """

        self._matrixRot=R

    def setTranslation(self,t):
        """
            Set translation vector

            :Parameters:
               - `t` (np.array(3,)) - a translation vector
        """
        self._translation=t

    def updateAxisFromRotation(self,R):
        """
            Update a rotation matrix

            :Parameters:
               - `R` (np.array(3,3) - a rotation matrix
        """
        self.m_axisX = R[:,0]
        self.m_axisY = R[:,1]
        self.m_axisZ = R[:,2]

        self._matrixRot = R

    def update(self,R,t):
        """
            Update both rotation matrix and translation vector

            :Parameters:
               - `R` (np.array(3,3) - a rotation matrix
               - `t` (np.array(3,)) - a translation vector
        """

        self.m_axisX = R[:,0]
        self.m_axisY = R[:,1]
        self.m_axisZ = R[:,2]
        self._translation = t
        self._matrixRot = R

    def addNode(self,nodeLabel,position, positionType="Global", desc =""):
        """
            Append a `Node` to a Frame

            :Parameters:
                - `nodeLabel` (str) - node label
                - `position` (np.array(3,)) - a translation vector
                - `positionType` (str) - two choice Global or Local

        """
        #TODO : use an Enum for the argment positionType

        logging.debug("new node (%s) added " % nodeLabel)

        isFind=False
        i=0
        for nodeIt in self._nodes:
            if str(nodeLabel+"_node") == nodeIt.m_name:
                isFind=True
                index = i
            i+=1

        if isFind:
            if positionType == "Global":
                self._nodes[index].m_global = position
                self._nodes[index].computeLocal(self._matrixRot,self._translation)
            elif positionType == "Local":
                self._nodes[index].m_local=position
                self._nodes[index].computeGlobal(self._matrixRot,self._translation)
            else :
                raise Exception("positionType not Known (Global or Local")

            logging.debug("[pyCGM2] node (%s) values updated"%(nodeLabel))
            previousDesc = self._nodes[index].m_desc
            if previousDesc != desc:
                logging.debug("[pyCGM2] node (%s) description updated [%s -> %s]"%(nodeLabel, previousDesc, desc))
                self._nodes[index].m_desc = desc

        else:
            node=Node(nodeLabel,desc=desc)
            if positionType == "Global":
                node.m_global=position
                node.computeLocal(self._matrixRot,self._translation)
            elif positionType == "Local":
                node.m_local=position
                node.computeGlobal(self._matrixRot,self._translation)
            else :
                raise Exception("positionType not Known (Global or Local")
            self._nodes.append(node)

    def getNode_byIndex(self,index):
        """
            Return a node within the list from its index

            :Parameters:
                - `index` (int) - index of the node within the list
            :Return:
                - `na` (pyCGM2.pyCGM2.Model.CGM2.Frame.Node) - a node instance
        """
        return self._nodes[index]

    def getNode_byLabel(self,label):
        """
            Return a node in the list from its label

         :Parameters:
            - `label` (str) - label of the node you want to find
         :Return:
            - `na` (pyCGM2.pyCGM2.Model.CGM2.Frame.Node) - a node instance
        """

        for nodeIt in self._nodes:
            if str(label+"_node") == nodeIt.m_name:
                logging.debug( " target label ( %s) - label find (%s) " %(label,nodeIt.m_name) )
                return nodeIt
            #else:
            #    raise Exception("Node label (%s) not found " %(label))

        return False




    def getNodeLabels(self,display=True ):
        """
            Display all node labels

        """
        labels=list()
        for nodeIt in self._nodes:
            labels.append(nodeIt.m_name[:-5])

            if display: print(nodeIt.m_name)

        return labels

    def eraseNodes(self):
        """
            erase all nodes
        """
        self._nodes=[]

    def getNodes(self):

        return self._nodes

    def isNodeExist(self,nodeLabel):

        flag = False
        for nodeIt in self._nodes:
            if str(nodeLabel+"_node") == nodeIt.m_name:
                logging.debug( " target label ( %s) - label find (%s) " %(nodeLabel,nodeIt.m_name) )
                flag = True
                break

        if not flag:
            logging.debug( " node label ( %s) doesn t exist " %(nodeLabel))

        return flag


    def getNodeIndex(self,nodeLabel):

        if self.isNodeExist(nodeLabel):

            for nodeIt in self._nodes:
                if str(nodeLabel+"_node") == nodeIt.m_name:
                    index = i
                    break
                i+=1
            return i
        else:
            raise Exception("[pyCGM2] node label doesn t exist" )


    def updateNode(self,nodeLabel,localArray,globalArray,desc=""):
        """
        update an existing node
        """

        index = self.getNodeIndex(nodeLabel)

        self._nodes[index].m_global = globalArray
        self._nodes[index].m_local = localArray
        self._nodes[index].m_desc = desc

    def copyNode(self,nodeLabel,nodeToCopy):

        indexToCopy = self.getNodeIndex(nodeToCopy)
        globalArray = self._nodes[indexToCopy].m_global
        localArray = self._nodes[indexToCopy].m_local
        desc = self._nodes[indexToCopy].m_desc

        if self.isNodeExist(nodeLabel):
            index =  self.getNodeIndex(nodeToCopy)
            self._nodes[index].m_global = globalArray
            self._nodes[index].m_local = localArray
            self._nodes[index].m_desc = desc
        else:
            self.addNode(nodeLabel,globalArray, position = "Global",desc =  desc)

    def getGlobalPosition(self,nodeLabel):

        node = self.getNode_byLabel(nodeLabel)

        return np.dot(self.getRotation(),node.getLocal())+ self.getTranslation()
