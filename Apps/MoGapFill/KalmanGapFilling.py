# -*- coding: utf-8 -*-
"""Nexus Operation : **KalmanGapFilling**

Low dimensional Kalman smoother that fills gaps in motion capture marker trajectories

This repository is a  Python implementation of a gap filling algorithm
(http://dx.doi.org/10.1016/j.jbiomech.2016.04.016)
that smooths trajectories in low dimensional subspaces, together with a Python plugin for Vicon Nexus.
"""


import sys
import ViconNexus
import numpy as np
import smooth


if __name__ == "__main__":

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED: # run Operation
        subject = NEXUS.GetSubjectNames()[0]
        print "Gap filling for subject ", subject

        markersLoaded = NEXUS.GetMarkerNames(subject) # nexus2.7 return all makers, even calibration only
        frames = NEXUS.GetFrameCount()

        markers =[]
        for i in range(0,len(markersLoaded)):
            data = NEXUS.GetTrajectory(subject,markersLoaded[i])
            if data != ([],[],[],[]):
                markers.append(markersLoaded[i])

        print "Populating data matrix"
        rawData = np.zeros((frames,len(markers)*3))
        for i in range(0,len(markers)):
            print i
            rawData[:,3*i-3], rawData[:,3*i-2], rawData[:,3*i-1], E = NEXUS.GetTrajectory(subject,markers[i])
            rawData[np.asarray(E)==0,3*i-3] = np.nan
            rawData[np.asarray(E)==0,3*i-2] = np.nan
            rawData[np.asarray(E)==0,3*i-1] = np.nan

        Y = smooth.smooth(rawData,tol =1e-2,sigR=1e-3,keepOriginal=True)
        print "Writing new trajectories"
        # Create new smoothed trjectories
        for i in range(0,len(markers)):
            E = np.ones((len(E),1)).tolist()
            NEXUS.SetTrajectory(subject,markers[i],Y[:,3*i-3].tolist(),Y[:,3*i-2].tolist(),Y[:,3*i-1].tolist(),E)
        print "Done"

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
