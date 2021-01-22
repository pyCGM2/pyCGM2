# -*- coding: utf-8 -*-
import logging
import numpy as np

from pyCGM2.Tools import  btkTools


#-------- EVENT PROCEDURES  ----------


# --- calibration procedure
class LowDimensionalKalmanFilterProcedure(object):
    """
        method from Burke et al. (Job 2016)
    """

    def __init__(self):
        self.description = "Burke (2016)"


    def _smooth(self,rawdata,tol=0.0025,sigR=1e-3,keepOriginal=True):

    	X = rawdata[~np.isnan(rawdata).any(axis=1)]

    	m = np.mean(X,axis=0)

    	logging.debug('Computing SVD...')

    	U, S, V = np.linalg.svd(X - m)

    	logging.debug('done')

    	d = np.nonzero(np.cumsum(S)/np.sum(S)>(1-tol))[0][0]

    	Q = np.dot(np.dot(V[0:d,:],np.diag(np.std(np.diff(X,axis=0),axis=0))),V[0:d,:].T)

    	logging.debug('Forward Pass')
    	state = []
    	state_pred = []
    	cov_pred = []
    	cov = []
    	cov.insert(0,1e12*np.eye(d))
    	state.insert(0,np.random.normal(0.0,1.0,d))
    	cov_pred.insert(0,1e12*np.eye(d))
    	state_pred.insert(0,np.random.normal(0.0,1.0,d))
    	for i in range(1,rawdata.shape[0]+1):

    		z =  rawdata[i-1,~np.isnan(rawdata[i-1,:])]
    		H = np.diag(~np.isnan(rawdata[i-1,:]))
    		H = H[~np.all(H == 0, axis=1)]
    		Ht = np.dot(H,V[0:d,:].T)

    		R = sigR*np.eye(H.shape[0])

    		state_pred.insert(i,state[i-1])
    		cov_pred.insert(i,cov[i-1] + Q)

    		K = np.dot(np.dot(cov_pred[i],Ht.T),np.linalg.inv(np.dot(np.dot(Ht,cov_pred[i]),Ht.T) + R))

    		state.insert(i,state_pred[i] + np.dot(K,(z - (np.dot(Ht,state_pred[i])+np.dot(H,m)))))
    		cov.insert(i,np.dot(np.eye(d) - np.dot(K,Ht),cov_pred[i]))

    	logging.debug('Backward Pass')
    	y = np.zeros(rawdata.shape)
    	y[-1,:] = np.dot(V[0:d,:].T,state[-1]) + m
    	for i in range(len(state)-2,0,-1):
    		state[i] =  state[i] + np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(state[i+1] - state_pred[i+1]))
    		cov[i] =  cov[i] + np.dot(np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(cov[i+1] - cov_pred[i+1])),cov[i])

    		y[i-1,:] = np.dot(V[0:d,:].T,state[i]) + m

    	if (keepOriginal):
    		y[~np.isnan(rawdata)] = rawdata[~np.isnan(rawdata)]

    	return y


    def fill(self,acq):
        logging.info("----LowDimensionalKalmanFilter gap filling----")
        btkmarkersLoaded  = btkTools.GetMarkerNames(acq)
        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()
        pfn = acq.GetPointFrameNumber()

        btkmarkers =[]
        for ml in btkmarkersLoaded:
            if btkTools.isPointExist(acq,ml) :
                btkmarkers.append(ml)

        # --------

        logging.debug("Populating data matrix")
        rawDatabtk = np.zeros((pfn,len(btkmarkers)*3))
        for i in range(0,len(btkmarkers)):
            values = acq.GetPoint(btkmarkers[i]).GetValues()
            residualValues = acq.GetPoint(btkmarkers[i]).GetResiduals()
            rawDatabtk[:,3*i-3] = values[:,0]
            rawDatabtk[:,3*i-2] = values[:,1]
            rawDatabtk[:,3*i-1] = values[:,2]
            E = residualValues[:,0]
            rawDatabtk[np.asarray(E)==-1,3*i-3] = np.nan
            rawDatabtk[np.asarray(E)==-1,3*i-2] = np.nan
            rawDatabtk[np.asarray(E)==-1,3*i-1] = np.nan

        Y2 = self._smooth(rawDatabtk,tol =1e-2,sigR=1e-3,keepOriginal=True)

        logging.debug("writing trajectories")

        # Create new smoothed trjectories
        filledMarkers  = list()
        for i in range(0,len(btkmarkers)):
            targetMarker = btkmarkers[i]
            if btkTools.isGap(acq,targetMarker):
                logging.info("marker (%s) --> filled"%(targetMarker))
                filledMarkers.append(targetMarker)
                val_final = np.zeros((pfn,3))
                val_final[:,0] = Y2[:,3*i-3]
                val_final[:,1] = Y2[:,3*i-2]
                val_final[:,2] = Y2[:,3*i-1]
                btkTools.smartAppendPoint(acq,targetMarker,val_final)
        logging.info("----LowDimensionalKalmanFilter gap filling [complete]----")

        return acq, filledMarkers


class GapFillingFilter(object):
    """

    """
    def __init__(self,procedure,acq):
        """
            :Parameters:
        """

        self.m_aqui = acq
        self.m_procedure = procedure

        self.filledMarkers  = None
        self.filledAcq  = None

    def getFilledAcq(self):
        return self.filledAcq

    def getFilledMarkers(self):
        return self.filledMarkers


    def fill(self):
        """

        """
        filledAcq,filledMarkers = self.m_procedure.fill(self.m_aqui)


        self.filledMarkers  = filledMarkers
        self.filledAcq  = filledAcq
