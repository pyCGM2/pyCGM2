# -*- coding: utf-8 -*-

import sys
import pyCGM2
import ViconNexus
import numpy as np
from pyCGM2.Tools import btkTools
from pyCGM2.Nexus import nexusTools
import matplotlib.pyplot as plt
from pyCGM2.Gap import gapFilling

def smooth(rawdata,tol=0.0025,sigR=1e-3,keepOriginal=True):

	X = rawdata[~np.isnan(rawdata).any(axis=1)]

	m = np.mean(X,axis=0)

	print 'Computing SVD...'

	U, S, V = np.linalg.svd(X - m)

	print 'done'

	d = np.nonzero(np.cumsum(S)/np.sum(S)>(1-tol))[0][0]

	Q = np.dot(np.dot(V[0:d,:],np.diag(np.std(np.diff(X,axis=0),axis=0))),V[0:d,:].T)

	print 'Forward Pass'
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

	print 'Backward Pass'
	y = np.zeros(rawdata.shape)
	y[-1,:] = np.dot(V[0:d,:].T,state[-1]) + m
	for i in range(len(state)-2,0,-1):
		state[i] =  state[i] + np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(state[i+1] - state_pred[i+1]))
		cov[i] =  cov[i] + np.dot(np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(cov[i+1] - cov_pred[i+1])),cov[i])

		y[i-1,:] = np.dot(V[0:d,:].T,state[i]) + m

	if (keepOriginal):
		y[~np.isnan(rawdata)] = rawdata[~np.isnan(rawdata)]

	return y

class KalmanMoGapFilled_Test():

    @classmethod
    def nexus_x2d(cls):

        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\gapFilling\\gaitWithGaps_withX2d\\"
        filenameNoExt = "gait_GAP"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

        acq_filled = btkTools.smartReader(str(DATA_PATH+"gait_GAP.-moGap.c3d"))


        subject = NEXUS.GetSubjectNames()[0]
        print "Gap filling for subject ", subject

        markersLoaded = NEXUS.GetMarkerNames(subject) # nexus2.7 return all makers, even calibration only
        frames = NEXUS.GetFrameCount()

        markers =[]
        for i in range(0,len(markersLoaded)):
            data = NEXUS.GetTrajectory(subject,markersLoaded[i])
            if data != ([],[],[],[]):
                markers.append(markersLoaded[i])

        #---------
        acq = btkTools.smartReader(str(DATA_PATH+filenameNoExt+".c3d"))
        btkmarkersLoaded  = btkTools.GetMarkerNames(acq)
        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()
        pfn = acq.GetPointFrameNumber()

        btkmarkers =[]
        for ml in btkmarkersLoaded:
            if btkTools.isPointExist(acq,ml) :
                btkmarkers.append(ml)
        #---------




        print "Populating data matrix"
        rawData = np.zeros((frames,len(markers)*3))
        for i in range(0,len(markers)):
            print i
            rawData[:,3*i-3], rawData[:,3*i-2], rawData[:,3*i-1], E = NEXUS.GetTrajectory(subject,markers[i])
            rawData[np.asarray(E)==0,3*i-3] = np.nan
            rawData[np.asarray(E)==0,3*i-2] = np.nan
            rawData[np.asarray(E)==0,3*i-1] = np.nan

        Y = smooth(rawData,tol =1e-2,sigR=1e-3,keepOriginal=True)
        print "Writing new trajectories"
        # Create new smoothed trjectories
        for i in range(0,len(markers)):
            if markers[i] =="LTHAD":
                E = np.ones((len(E),1)).tolist()
                val0 = Y[:,3*i-3].tolist()
                val1 = Y[:,3*i-2].tolist()
                val2 = Y[:,3*i-1].tolist()
            #NEXUS.SetTrajectory(subject,markers[i],Y[:,3*i-3].tolist(),Y[:,3*i-2].tolist(),Y[:,3*i-1].tolist(),E)
        print "Done"



        # --------
        print "Populating data matrix"
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

        Y2 = smooth(rawDatabtk,tol =1e-2,sigR=1e-3,keepOriginal=True)
        print "Writing new trajectories"
        # Create new smoothed trjectories
        for i in range(0,len(btkmarkers)):
            targetMarker = btkmarkers[i]
            if btkTools.isGap(acq,targetMarker):
                val_final = np.zeros((pfn,3))
                val_final[:,0] = Y2[:,3*i-3]
                val_final[:,1] = Y2[:,3*i-2]
                val_final[:,2] = Y2[:,3*i-1]
                btkTools.smartAppendPoint(acq,targetMarker,val_final)
                nexusTools.setTrajectoryFromAcq(NEXUS,subject,targetMarker,acq)
        print "Done"

    @classmethod
    def nexus_noX2d(cls):

        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\gapFilling\\gaitWithGaps_noX2d\\"
        filenameNoExt = "gait_GAP"
        NEXUS.OpenTrial( str(DATA_PATH+filenameNoExt), 30 )

        acq_filled = btkTools.smartReader(str(DATA_PATH+"gait_GAP.-moGap.c3d"))


        subject = NEXUS.GetSubjectNames()[0]
        print "Gap filling for subject ", subject

        markersLoaded = NEXUS.GetMarkerNames(subject) # nexus2.7 return all makers, even calibration only
        frames = NEXUS.GetFrameCount()

        markers =[]
        for i in range(0,len(markersLoaded)):
            data = NEXUS.GetTrajectory(subject,markersLoaded[i])
            if data != ([],[],[],[]):
                markers.append(markersLoaded[i])

        #---------
        acq = btkTools.smartReader(str(DATA_PATH+filenameNoExt+".c3d"))
        btkmarkersLoaded  = btkTools.GetMarkerNames(acq)
        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()
        pfn = acq.GetPointFrameNumber()

        btkmarkers =[]
        for ml in btkmarkersLoaded:
            if btkTools.isPointExist(acq,ml) :
                btkmarkers.append(ml)
        #---------




        print "Populating data matrix"
        rawData = np.zeros((frames,len(markers)*3))
        for i in range(0,len(markers)):
            print i
            rawData[:,3*i-3], rawData[:,3*i-2], rawData[:,3*i-1], E = NEXUS.GetTrajectory(subject,markers[i])
            rawData[np.asarray(E)==0,3*i-3] = np.nan
            rawData[np.asarray(E)==0,3*i-2] = np.nan
            rawData[np.asarray(E)==0,3*i-1] = np.nan

        Y = smooth(rawData,tol =1e-2,sigR=1e-3,keepOriginal=True)
        print "Writing new trajectories"
        # Create new smoothed trjectories
        for i in range(0,len(markers)):
            if markers[i] =="LTHAD":
                E = np.ones((len(E),1)).tolist()
                val0 = Y[:,3*i-3].tolist()
                val1 = Y[:,3*i-2].tolist()
                val2 = Y[:,3*i-1].tolist()
            #NEXUS.SetTrajectory(subject,markers[i],Y[:,3*i-3].tolist(),Y[:,3*i-2].tolist(),Y[:,3*i-1].tolist(),E)
        print "Done"



        # --------
        print "Populating data matrix"
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

        Y2 = smooth(rawDatabtk,tol =1e-2,sigR=1e-3,keepOriginal=True)
        print "Writing new trajectories"
        # Create new smoothed trjectories
        for i in range(0,len(btkmarkers)):
            targetMarker = btkmarkers[i]
            if btkTools.isGap(acq,targetMarker):
                val_final = np.zeros((pfn,3))
                val_final[:,0] = Y2[:,3*i-3]
                val_final[:,1] = Y2[:,3*i-2]
                val_final[:,2] = Y2[:,3*i-1]
                btkTools.smartAppendPoint(acq,targetMarker,val_final)
                nexusTools.setTrajectoryFromAcq(NEXUS,subject,targetMarker,acq)
        print "Done"

        plt.plot(acq.GetPoint("LTIB").GetValues(),"or")
        plt.plot(acq_filled.GetPoint("LTIB").GetValues(),"-g")
        plt.show()

    @classmethod
    def gapFilterTest(cls):

        DATA_PATH ="C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\operations\\gapFilling\\gaitWithGaps_noX2d\\"
        acq = btkTools.smartReader(str(DATA_PATH+"gait_GAP.c3d"))
        acq_filled = btkTools.smartReader(str(DATA_PATH+"gait_GAP.-moGap.c3d"))


        gfp =  gapFilling.LowDimensionalKalmanFilterProcedure()
        gff = gapFilling.GapFillingFilter(gfp,acq)
        gff.fill()
        filledAcq  = gff.getFilledAcq()
        filledMarkers  = gff.getFilledMarkers()
        for marker in filledMarkers:
            plt.plot(filledAcq.GetPoint(marker).GetValues(),"or")
            plt.plot(acq_filled.GetPoint(marker).GetValues(),"-g")
            plt.show()


if __name__ == "__main__":

    #KalmanMoGapFilled_Test.nexus_x2d()
    #KalmanMoGapFilled_Test.nexus_noX2d()
    KalmanMoGapFilled_Test.gapFilterTest()
