"""
The module contains procedures for detecting foot contact event.

check out the script : *\Tests\test_events.py* for examples
"""

from typing import List, Tuple, Dict, Optional,Union
import pyCGM2; LOGGER = pyCGM2.LOGGER

import btk
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn import preprocessing
import threading

try:
    import onnxruntime as rt
except ImportError as e:
    LOGGER.logger.error(f"ImportError: {e}")
    LOGGER.logger.error("[pyCGM2] - onnxruntime encounter an issue. \
                        if your message mentions a dll issue, \ " \
                        "replace pyCGM2.exe by the rullThemAllCommands.py located in the Apps/Commands folder of your pyCGM2 ")
    
    pass


from pyCGM2.Events import eventProcedures


# --- abstract procedure
class EventProcedure(object):
    """
    Abstract class for event procedures.

    This class serves as a foundation for specific event detection procedures in gait analysis. 
    It should be extended to implement methods for detecting specific types of gait events.
    """
    def __init__(self):
        pass


#-------- EVENT PROCEDURES  ----------



        
class IntellEventProcedure(eventProcedures.EventProcedure):
    """
    Gait event detection procedure based on Intellevent

    """

    def __init__(self,progressionAxis,forwardProgression, min_peak_threshold = 0.3, distance = 25):
        """
        Initializes the IntellEventProcedure class.
        """
        super(IntellEventProcedure, self).__init__()
        self.description = "intellevent"

        self.m_progressionAxis = progressionAxis
        self.m_forwardProgression = forwardProgression

        self.m_markers = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"]

        self.m_min_peak_threshold = min_peak_threshold
        self.m_distance = distance
        
        self._baseFrequency= 150

        self.m_model_file_fc = None
        self.m_model_file_fo = None

    def setModels(self,trainedModelInitialContactFile, trainedModelFootOffFile):

        if os.path.isfile(trainedModelInitialContactFile):          
            self.m_model_file_fc = trainedModelInitialContactFile
        else: 
            LOGGER.logger.error("[intellevent] - initial Contact trained model not exist. check your path")
            raise FileNotFoundError ("[intellevent] - initial Contact trained model not exist. check your path") 


        if os.path.isfile(trainedModelFootOffFile):
            self.m_model_file_fo = trainedModelFootOffFile
        else:
            LOGGER.logger.error("[intellevent] - foot off trained model not exist. check your path")
            raise FileNotFoundError ("[intellevent] - foot off trained model not exist. check your path")


    def __prepare(self,acq):
      
        
        x_traj, y_traj, z_traj = [], [], []


        # Get the corresponding index for each marker name in 'marker_list'
        for marker_name in self.m_markers:
            xyz = acq.GetPoint(marker_name).GetValues() 
            x_traj.append(xyz[:,0])
            y_traj.append(xyz[:,1])
            z_traj.append(xyz[:,2])


       
        # The current best model uses the x and z axis velocity for the IC model
        # and the x, y, and z axis velocity for the FO model
        # first component should be forward axis progression
        if self.m_progressionAxis == "X": # np.mean(np.abs(prog_x)) > np.mean(np.abs(prog_y)):
            ic_traj = np.concatenate([x_traj, z_traj])
            fo_traj = np.concatenate([x_traj, y_traj, z_traj])
        else:
            ic_traj = np.concatenate([y_traj, z_traj])
            fo_traj = np.concatenate([y_traj, x_traj, z_traj])

        LOGGER.logger.info("[intellevent] - prepare data----> done")
        return ic_traj,fo_traj

    def __preprocess(self,acq,ic_traj,fo_traj):
        def reshape_data(traj):
            """
            Reshape a 2D numpy array into the required input format for the neural network.

            Parameters:
                traj (numpy.ndarray): A 2D array of shape (num_features, num_frames).

            Returns:
                numpy.ndarray: Reshaped array of shape (num_samples, num_frames, num_features).
            """
            rs_traj = np.transpose(np.array(traj).reshape(1, traj.shape[0], traj.shape[1]), (0, 2, 1))
            return rs_traj

        def resample_data(traj, sample_frequ, frequ_to_sample):
            """
            Resample the data to the desired frequency.
            """
            period = '{}N'.format(int(1e9 / sample_frequ))
            index = pd.date_range(0, periods=len(traj[0, :]), freq=period)
            resampled_data = [pd.DataFrame(val, index=index).resample('{}N'.format(int(1e9 / frequ_to_sample))).mean() for val
                            in traj]
            resampled_data = [np.array(traj.interpolate(method='linear')) for traj in resampled_data]
            resampled_data = np.concatenate(resampled_data, axis=1)
            return resampled_data
        
        # x and y-coordinates need to be standardized depending on the starting direction,
        # z coordinates are always the same
        if any(ic_traj[0, 0:10] < 0) or any(ic_traj[3, 0:10] < 0):
            ic_traj[0:6, :] = (ic_traj[0:6, :] - np.mean(ic_traj[0:6, :], axis=1).reshape(6,1)) * (-1)
            fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)) * (-1)


        # calculate the first derivative (= velocity) of the trajectories
        ic_velo = np.gradient(ic_traj, axis=1)
        fo_velo = np.gradient(fo_traj, axis=1)

        # standardize between 0.1 and 1.1 for the machine learning algorithm (zeros will be ignored!)
        ic_velo = preprocessing.minmax_scale(ic_velo, feature_range=(0.1, 1.1), axis=1)
        fo_velo = preprocessing.minmax_scale(fo_velo, feature_range=(0.1, 1.1), axis=1)

        #Down / up sampling?
        cam_frequency = acq.GetPointFrequency()
        if cam_frequency != self._baseFrequency:
            rs_ic_velo = resample_data(ic_velo, cam_frequency, self._baseFrequency).transpose()
            rs_fo_velo = resample_data(fo_velo, cam_frequency, self._baseFrequency).transpose()
        else:
            rs_ic_velo = ic_velo
            rs_fo_velo = fo_velo

        # both 'ic_velo' and 'fo_velo' should be in the shape (num_features, num_frames) (e.g. (12, 500) or (18, 500))
        # for the prediction we need the shape of (num_samples, num_frames, num_features)
        # num_samples = 1, num_frames = length of trial (e.g. 500), num_features = velocity of trajectories (e.g. 12 or 18)
        # check with rs_ic_velo.shape
        rs_ic_velo = reshape_data(rs_ic_velo) #rs_ic_velo
        rs_fo_velo = reshape_data(rs_fo_velo) #rs_fo_velo

        LOGGER.logger.info("[intellevent] - preprocess data----> done")

        return rs_ic_velo, rs_fo_velo


    def __predict(self,fc_data,fo_data ):

        if  self.m_model_file_fc is None or self.m_model_file_fo is None:   
            LOGGER.logger.error("[intellevent] - no models provided. please use the setModels methods")
            raise 
        else:
            providers = ['CPUExecutionProvider']
            model_fc= rt.InferenceSession(self.m_model_file_fc, providers=providers)
            ic_preds = model_fc.run(['time_distributed'], {"input_1": fc_data.tolist()})
    
        
            model_fo = rt.InferenceSession(self.m_model_file_fo , providers=providers)
            fo_preds = model_fo.run(['time_distributed'], {"input_1": fo_data.tolist()})

            return ic_preds[0][0], fo_preds[0][0]


    def detect(self,acq:btk.btkAcquisition)-> Union[Tuple[int, int, int, int], int] :
        """
        Detect events using the intellevent.

        Args:
            acq (btk.btkAcquisition): A BTK acquisition instance containing motion capture data.

        Returns:
            Union[Tuple[int, int, int, int], int]: Frames indicating the left foot strike, left foot off, 
                                                   right foot strike, and right foot off respectively. 
                                                   Returns 0 if detection fails.
        """


        ff = acq.GetFirstFrame()
        lhee = acq.GetPoint("LHEE").GetValues()
        rhee = acq.GetPoint("RHEE").GetValues()

        LOGGER.logger.info("[intellevent] - prepare data")
        ic_traj,fo_traj = self.__prepare(btk.btkAcquisition.Clone(acq))
        LOGGER.logger.info("[intellevent] - preprocess data")
        fc_data,fo_data = self.__preprocess(acq,ic_traj,fo_traj)
        
        LOGGER.logger.info("[intellevent] - predict data")
        fc_preds,fo_preds = self.__predict(fc_data,fo_data)


        indexes_fs_left=[]
        indexes_fo_left=[]
        indexes_fs_right =[]
        indexes_fo_right=[]

        [loc, height] = find_peaks(fc_preds[:, 1], height=self.m_min_peak_threshold, 
                                   distance=self.m_distance)

        loc = np.ceil( (loc / self._baseFrequency) * acq.GetPointFrequency())
        loc = [int(it) for it in loc]


        if self.m_progressionAxis== "X":axis = 0
        if self.m_progressionAxis== "Y":axis = 1

        for index in loc:
            if self.m_forwardProgression:
                if lhee[index,axis] > rhee[index,axis]:
                    indexes_fs_left.append(index+ff)
                else:
                    indexes_fs_right.append(index+ff)
            else:
                if lhee[index,axis] < rhee[index,axis]:
                    indexes_fs_left.append(index+ff)
                else:
                    indexes_fs_right.append(index+ff)        

        [loc, height] = find_peaks(fo_preds[:, 1], 
                                   height=self.m_min_peak_threshold, 
                                   distance=self.m_distance)
        loc = np.ceil( (loc / self._baseFrequency) * acq.GetPointFrequency())
        loc = [int(it) for it in loc]

        for index in loc:
            if self.m_forwardProgression:
                if lhee[index,axis] < rhee[index,axis]:
                    indexes_fo_left.append(index+ff)
                else:
                    indexes_fo_right.append(index+ff)
            else: 
                if lhee[index,axis] > rhee[index,axis]:
                    indexes_fo_left.append(index+ff)
                else:
                    indexes_fo_right.append(index+ff)

        
        return indexes_fs_left,indexes_fo_left, indexes_fs_right, indexes_fo_right        
    