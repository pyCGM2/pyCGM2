# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 11:05:55 2015

@author: fleboeuf
"""

import numpy as np
import pdb
import logging

def setFrameData(a1,a2,sequence):
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

class Frame(object):
    """ a Frame is an axis system
 
    .. note:: Frame embbeds Node object   

    .. todo:: rename Frame by SegmentalFrame ??   
        
    """    
    
    
    class Node(object):
        """ A node is a local frame point. a node is characterized by a constant position
        """        
        def __init__(self,label):
            """ Constructor
            
            Generally a node come from a global position.  
            
           :Parameters:
               - `label` (str) - label of the node you want to build 
           
           .. warning:: automatically the suffixe "_node" will be added to the label
    
           .. todo:: wow to deal with muscle insertion

            """

            self.m_name = label+"_node"
            self.m_global = np.zeros((1,3))
            self.m_local = np.zeros((1,3))
            
        def computeLocal(self,rot,t):
            """
            compute local position from global
            
           :Parameters:
               - `rot` (np.array((3,3))) - a rotation matrix 
               - `t` (np.array((1,3))) - a translation vector 
               
            """
            self.m_local=np.dot(rot.T,(self.m_global-t))
    
        def computeGlobal(self,rot,t):
            """
            compute global position from local
            
           :Parameters:
               - `rot` (np.array((3,3))) - a rotation matrix 
               - `t` (np.array((1,3))) - a translation vector 
           
            """

            self.m_global=np.dot(rot,self.m_local) +t
    
    def __init__(self):
        """ Constructor
        Initialization of both X-Y-Z axes, translation vector, rotation matrix and node list 
        """
        self.m_axisX=np.zeros((1,3))                    
        self.m_axisY=np.zeros((1,3))
        self.m_axisZ=np.zeros((1,3))
        
        self._translation=np.zeros((1,3))
        self._matrixRot=np.zeros((3,3))


        self._nodes=[]

    def getRotation(self):
        """ get rotation matrix
        """
        return self._matrixRot

    def getTranslation(self):
        """ get translation vector
        """
        return self._translation

    def setRotation(self, R):
        """ set rotation matrix
        
         :Parameters:
               - `rot` (np.array((3,3))) - a rotation matrix 
        """

        self._matrixRot=R

    def setTranslation(self,t):
        """ set translation vector
        
         :Parameters:
               - `t` (np.array((3,))) - a translation vector 
        """
        self._translation=t

    def updateAxisFromRotation(self,R):
        self.m_axisX = R[:,0]
        self.m_axisY = R[:,1]
        self.m_axisZ = R[:,2]

        self._matrixRot = R

    def update(self,R,t):
        self.m_axisX = R[:,0]
        self.m_axisY = R[:,1]
        self.m_axisZ = R[:,2]
        self._translation = t
        self._matrixRot = R

    def addNode(self,nodeLabel,position, positionType="Global"):
        """ update or append a node 
         
         caution with the argument positionType
        
         :Parameters:
            - `nodeLabel` (str) - node of the label you want to add 
            - `position` (np.array((3,))) - a translation vector 
            - `positionType` (str) - two choice Global or Local 

        .. todo : use an Enum for the argment positionType
        """
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
            
        else:
            node=Frame.Node(nodeLabel)
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
        """ return a node in the list from its index
        
         :Parameters:
            - `index` (int) - index in the list
        """
        return self._nodes[index]           
    
    def getNode_byLabel(self,label):
        """ return a node in the list from its label
        
         :Parameters:
            - `label` (str) - label of the node you want to find
        """
        
        for nodeIt in self._nodes:
            if str(label+"_node") == nodeIt.m_name:
                logging.debug( " target label ( %s) - label find (%s) " %(label,nodeIt.m_name) )       
                return nodeIt
        
        return False        




    def printAllNodes(self):
        """ print the label of nodes
    
        """
        for nodeIt in self._nodes:
            print nodeIt.m_name

    def eraseNodes(self):    
        self._nodes=[]

