# coding: utf-8
import logging
import numpy as np
from numpy import matlib as mb
from scipy import signal
from scipy.signal import argrelextrema
import pyCGM2
from keras.models import model_from_json
try:
    from pyCGM2 import btk
except:
    logging.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk

from pyCGM2.Processing import progressionFrame
from pyCGM2.Tools import btkTools

class DeepEventProcedure(object):
    """
        Gait Event detection from deepEvent
    """

    def __init__(self):
        self.description = "Deepevent(2019)"

    def __predict(self,load_model,acq,markers,pfn,freq):

        def derive_centre(marker,pfn,freq):
            # Compute velocity
            marker_der = (marker[2:pfn,:] - marker[0:(pfn-2),:]) / (2 / freq)
            marker_der = np.concatenate(([[0,0,0]],marker_der,[[0,0,0]]),axis=0)
            return marker_der

        def filter(acq,marker,fc):
            # Butterworth filter
            b, a = signal.butter(4, fc/(acq.GetPointFrequency()/2))
            Mean = np.mean(marker,axis=0)
            Minput = marker - mb.repmat(Mean,acq.GetPointFrameNumber(),1)
            Minput = signal.filtfilt(b,a,Minput,axis=0)
            Moutput = Minput + np.matlib.repmat(Mean,acq.GetPointFrameNumber(),1)
            return Moutput

        nframes = 1536
        nb_data_in = 36 #6 markers x 3

        inputs = np.zeros((1,nframes,nb_data_in))
        for k in range(6):
            values = acq.GetPoint(markers[k]).GetValues()
            inputs[0,0:pfn,k*3: (k + 1)*3] = filter(acq,values,6)
            inputs[0,0:pfn,3 * len(markers) + k*3:3 * len(markers) +  (k + 1)*3] = derive_centre(inputs[0,:,k * 3:(k+1)*3],pfn,freq)

        # Prediction with the model
        predicted = load_model.predict(inputs) #shape[1,nb_frames,5] 0: no event, 1: Left Foot Strike, 2: Right Foot Strike, 3:Left Toe Off, 4: Right Toe Off

        #Threshold to set the gait events
        predicted_seuil = predicted
        for j in range(nframes):
            if predicted[0,j,1] <= 0.01:
                predicted_seuil[0,j,1] = 0
            if predicted[0,j,2] <= 0.01:
                predicted_seuil[0,j,2] = 0
            if predicted[0,j,3] <= 0.01:
                predicted_seuil[0,j,3] = 0
            if predicted[0,j,4] <= 0.01:
                predicted_seuil[0,j,4] = 0

        predicted_seuil_max = np.zeros((1,nframes,5))
        for j in range(1,5):
            predicted_seuil_max[0,argrelextrema(predicted_seuil[0,:,j],np.greater)[0],j] = 1

        for j in range(nframes):
            if np.sum(predicted_seuil_max[0,j,:]) == 0: predicted_seuil_max[0,j,0] = 1

        eventLFS = np.argwhere(predicted_seuil_max[0,:,1])
        eventRFS = np.argwhere(predicted_seuil_max[0,:,2])
        eventLFO = np.argwhere(predicted_seuil_max[0,:,3])
        eventRFO = np.argwhere(predicted_seuil_max[0,:,4])

        return eventLFS,eventRFS,eventLFO,eventRFO

    def detect(self,acq):
        """
        """
        with  open(pyCGM2.DEEPEVENT_DATA_PATH+'DeepEventModel.json','r') as json_file:
            loaded_model_json = json_file.read()


        model = model_from_json(loaded_model_json)
        model.load_weights(pyCGM2.DEEPEVENT_DATA_PATH+"DeepEventWeight.h5")

        acq.ClearEvents()

        acqF = btk.btkAcquisition.Clone(acq)
        pfn = acqF.GetPointFrameNumber()
        freq = acqF.GetPointFrequency()


        markers = ["LANK","RANK","LTOE","RTOE","LHEE","RHEE"]

        pfp = progressionFrame.PointProgressionFrameProcedure(marker="LANK")
        pff = progressionFrame.ProgressionFrameFilter(acq,pfp)
        pff.compute()

        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]

        btkTools.applyRotation(acq,markers,globalFrame,forwardProgression)

        eventLFS,eventLFO,eventRFS,eventRFO = self.predict(model,acq,markers,pfn,freq)


	    # for ind_indice in range(eventLFS.shape[0]):
	    #     newEvent=btk.btkEvent()
	    #     newEvent.SetLabel("Foot Strike")
	    #     newEvent.SetContext("Left")
	    #     newEvent.SetTime((ff-1)/freq + float(eventLFS[ind_indice]/freq))
	    #     newEvent.SetSubject(SubjectValue[0])
	    #     newEvent.SetId(1)
	    #     acqF.AppendEvent(newEvent)
		#
	    # for ind_indice in range(eventRFS.shape[0]):
	    #     newEvent=btk.btkEvent()
	    #     newEvent.SetLabel("Foot Strike")
	    #     newEvent.SetContext("Right")
	    #     newEvent.SetTime((ff-1)/freq + float(eventRFS[ind_indice]/freq))
	    #     newEvent.SetSubject(SubjectValue[0])
	    #     newEvent.SetId(1)
	    #     acqF.AppendEvent(newEvent)
		#
	    # for ind_indice in range(eventLFO.shape[0]):
	    #     newEvent=btk.btkEvent()
	    #     newEvent.SetLabel("Foot Off")
	    #     newEvent.SetContext("Left") #
	    #     newEvent.SetTime((ff-1)/freq + float(eventLFO[ind_indice]/freq))
	    #     newEvent.SetSubject(SubjectValue[0])
	    #     newEvent.SetId(2)
	    #     acqF.AppendEvent(newEvent)
		#
	    # for ind_indice in range(eventRFO.shape[0]):
	    #     newEvent=btk.btkEvent()
	    #     newEvent.SetLabel("Foot Off")
	    #     newEvent.SetContext("Right") #
	    #     newEvent.SetTime((ff-1)/freq + float(eventRFO[ind_indice]/freq))
	    #     newEvent.SetSubject(SubjectValue[0])
	    #     newEvent.SetId(2)
	    #     acqF.AppendEvent(newEvent)
		#
	    # save(acqF,filenameOut)
