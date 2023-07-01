# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Gap
#APIDOC["Draft"]=False
#--end--
"""
The module contains filter and procedure for filling gap

check out the script : *\Tests\test_gap.py* for examples
"""

import pyCGM2; LOGGER = pyCGM2.LOGGER
import numpy as np
import warnings


from pyCGM2.Tools import  btkTools

# --- abstract procedure
class GapFillingProcedure(object):
    def __init__(self):
        pass

# --- concrete procedure
class Burke2016KalmanGapFillingProcedure(GapFillingProcedure):
    """
        gap fill procedure according  Burke et al. (Job 2016)

        Burke, M.; Lasenby, J. (2016) Estimating missing marker positions using low dimensional Kalman smoothing. 
        In : Journal of biomechanics, vol.49, n°9, p.1854-1858. DOI: 10.1016/j.jbiomech.2016.04.016.
    """

    def __init__(self):
        super(Burke2016KalmanGapFillingProcedure, self).__init__()
        self.description = "Burke (2016)"


    def _smooth(self,rawdata,tol=0.0025,sigR=1e-3,keepOriginal=True):

        X = rawdata[~np.isnan(rawdata).any(axis=1)]
        m = np.mean(X,axis=0)
        LOGGER.logger.debug('Computing SVD...')
        U, S, V = np.linalg.svd(X - m)

        LOGGER.logger.debug('done')

        d = np.nonzero(np.cumsum(S)/np.sum(S)>(1-tol))[0][0]

        Q = np.dot(np.dot(V[0:d,:],np.diag(np.std(np.diff(X,axis=0),axis=0))),V[0:d,:].T)

        LOGGER.logger.debug('Forward Pass')
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

        LOGGER.logger.debug('Backward Pass')
        y = np.zeros(rawdata.shape)
        y[-1,:] = np.dot(V[0:d,:].T,state[-1]) + m
        for i in range(len(state)-2,0,-1):
            state[i] =  state[i] + np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(state[i+1] - state_pred[i+1]))
            cov[i] =  cov[i] + np.dot(np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(cov[i+1] - cov_pred[i+1])),cov[i])

            y[i-1,:] = np.dot(V[0:d,:].T,state[i]) + m

        if (keepOriginal):
            y[~np.isnan(rawdata)] = rawdata[~np.isnan(rawdata)]

        return y


    def fill(self,acq,**kwargs):
        """fill gap

        Args:
            acq (Btk.Acquisition): a btk acquisition instance

        """
        LOGGER.logger.info("----Burke2016 gap filling----")
        btkmarkersLoaded  = btkTools.GetMarkerNames(acq)
        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()
        pfn = acq.GetPointFrameNumber()

        if "markers" in kwargs:
            btkmarkers = kwargs["markers"]
        else:
            btkmarkers =[]
            for ml in btkmarkersLoaded:
                if btkTools.isPointExist(acq,ml) :
                    btkmarkers.append(ml)
        # --------

        LOGGER.logger.debug("Populating data matrix")
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

        LOGGER.logger.debug("writing trajectories")

        # Create new smoothed trjectories
        filledMarkers  = list()
        for i in range(0,len(btkmarkers)):
            targetMarker = btkmarkers[i]
            if btkTools.isGap(acq,targetMarker):
                LOGGER.logger.info("marker (%s) --> filled"%(targetMarker))
                filledMarkers.append(targetMarker)
                val_final = np.zeros((pfn,3))
                val_final[:,0] = Y2[:,3*i-3]
                val_final[:,1] = Y2[:,3*i-2]
                val_final[:,2] = Y2[:,3*i-1]
                btkTools.smartAppendPoint(acq,targetMarker,val_final)
        LOGGER.logger.info("----Burke2016 gap filling [complete]----")

        return acq, filledMarkers


class Gloersen2016GapFillingProcedure(GapFillingProcedure):
    """
    Fills gaps in the marker postion data exploiting intercorrelations between marker coordinates.


    See:
        Federolf PA. 2013. PLoS ONE 8(10):e78689. doi:10.1371/journal.pone.0078689
        Gløersen Ø, Federolf P. 2016. PLoS ONE 11(3):e0152616. doi:10.1371/journal.pone.0152616

    """

    def __init__(self):
        super(Gloersen2016GapFillingProcedure, self).__init__()
        self.description = "Gløersen Ø, Federolf P. (2016)"



    def _distance2marker(self, data, ix_channels_with_gaps):
            """Computes the Euclidean distance for each marker with missing data to each other marker.

            Parameters
            ----------
            data : (N, M) array_like
                The marker position data with N time steps across M channels.
            ix_channels_with_gaps : array_lik
                The indexes of the channels with missing marker data.
            
            Returns
            -------
            weights : (M'', M') array_like
                Array of pair-wise Euclidean distances between markers with gaps and each other marker.
                Here, M'' is the number of markers with missing data, and M' is the number of markers.
            """
            from scipy.spatial.distance import cdist

            # Get shape of data
            N, M = data.shape
            
            # Reshape data to shape (3, n_markers, n_time_steps)
            ix_markers_with_gaps = ( ix_channels_with_gaps[2::3] // 3 )  # columns of markers with gaps
            n_markers_with_gaps = len(ix_markers_with_gaps)
            data_reshaped = (data.T).reshape((3, M//3, N), order="F")

            # Compute weights based on distances
            weights = np.empty((n_markers_with_gaps, M//3, N))
            for i in range(N):
                weights[:,:,i] = cdist(data_reshaped[:,ix_markers_with_gaps,i].T, data_reshaped[:,:,i].T, "euclidean")
            weights = np.nanmean(weights, axis=-1)
            return weights
        
    def _PCA(self,data):
        """Performs principal components analysis by means of singular value decomposition.

        Parameters
        ----------
        data : (N, M) array_like
            The marker position data with N time steps across M channels.
        
        Returns
        -------
        PC : 
            The principal component vectors.
        sqrtEV : 
            The square root of the eigenvalues.
        """

        # Get shape of data
        N, M = data.shape

        # Calculate Y matrix
        Y = data / np.sqrt(N-1)

        # Find principal components
        _, sqrtEV, VT = np.linalg.svd(Y, full_matrices=0)
        PC = VT.T
        return PC, sqrtEV

    def _reconstruct(self,data, weight_scale, mm_weight, min_cum_sv):
        """Reconstructs missing marker data using the strategy based on intercorrelations between marker clusters.

        Parameters
        ----------
        data : (N, M) array_like
            Array of marker position data with N time steps across M channels.
            Note that the mean trajectories have been subtracted from the original marker position data,
            as to obtain a coordinate system moving with the subject.

        Returns
        -------
        reconstructed_data : (N, M) array_like
            The data with reconstructed marker trajectories.
        """

        # Get shape of data
        n_time_steps, n_channels = data.shape

        # Find channels with missing data
        ix_channels_with_gaps, = np.nonzero(np.any(np.isnan(data), axis=0))
        ix_time_steps_with_gaps, = np.nonzero(np.any(np.isnan(data), axis=1))

        # Compute the weights
        weights = self._distance2marker(data, ix_channels_with_gaps)
        if weights.shape[0] >= 1:
            weights = np.min(weights, axis=0)
        weights = np.exp(-np.divide(weights**2, 2*weight_scale**2))
        weights[ix_channels_with_gaps[2::3]//3] = mm_weight

        # Define matrices need for reconstruction
        M_zeros = data.copy()
        M_zeros[:,ix_channels_with_gaps] = 0
        N_no_gaps = np.delete(data, ix_time_steps_with_gaps, axis=0)
        N_zeros = N_no_gaps.copy()
        N_zeros[:,ix_channels_with_gaps] = 0

        # Normalize matrices to unit variance, then multiply by weights
        mean_N_no_gaps = np.mean(N_no_gaps, axis=0)
        mean_N_zeros = np.mean(N_zeros, axis=0)
        stdev_N_no_gaps = np.std(N_no_gaps, axis=0)
        stdev_N_no_gaps[np.argwhere(stdev_N_no_gaps==0)[:,0]] = 1

        M_zeros = np.divide(( M_zeros - np.tile(mean_N_zeros.reshape(-1,1).T, (M_zeros.shape[0], 1)) ), \
            np.tile(stdev_N_no_gaps.reshape(-1,1).T, (M_zeros.shape[0],1))) * \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (M_zeros.shape[0],1))
        
        N_no_gaps = np.divide(( N_no_gaps - np.tile(mean_N_no_gaps.reshape(-1,1).T, (N_no_gaps.shape[0],1)) ), \
            np.tile(stdev_N_no_gaps.reshape(-1,1).T, (N_no_gaps.shape[0],1))) * \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (N_no_gaps.shape[0],1))
        
        N_zeros = np.divide(( N_zeros - np.tile(mean_N_zeros.reshape(-1,1).T, (N_zeros.shape[0],1)) ), \
            np.tile(stdev_N_no_gaps.reshape(-1,1).T, (N_zeros.shape[0],1))) * \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (N_zeros.shape[0],1))

        # Calculate the principal component vectors and the eigenvalues
        PC_vectors_no_gaps, sqrtEV_no_gaps = self._PCA(N_no_gaps)
        PC_vectors_zeros, sqrtEV_zeros = self._PCA(N_zeros)

        # Determine the number of eigenvectors to consider
        n_eigvecs = np.max([np.argwhere(np.cumsum(sqrtEV_no_gaps) >= min_cum_sv*np.sum(sqrtEV_no_gaps))[:,0][0], \
            np.argwhere(np.cumsum(sqrtEV_zeros) >= min_cum_sv*np.sum(sqrtEV_zeros))[:,0][0]])
        PC_vectors_no_gaps = PC_vectors_no_gaps[:,:n_eigvecs+1]
        PC_vectors_zeros = PC_vectors_zeros[:,:n_eigvecs+1]

        # Calculate the transformation matrix
        T = PC_vectors_no_gaps.T @ PC_vectors_zeros

        # Calculate the reconstruction matrix, see: Federolf (2013).
        R = M_zeros @ PC_vectors_zeros @ T @ PC_vectors_no_gaps.T

        # Reverse the normalization
        R = np.tile(mean_N_no_gaps.reshape(-1,1).T, (data.shape[0],1)) + \
            np.divide(R * np.tile(stdev_N_no_gaps.reshape(-1,1).T, (data.shape[0],1)), \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (data.shape[0],1)))
        
        # Replace missing data with reconstructed data
        reconstructed_data = data.copy()
        for ix in ix_channels_with_gaps:
            reconstructed_data[:,ix] = R[:,ix]
        return reconstructed_data


    def fill(self,acq, **kwargs):
        """Fills gaps in the marker postion data exploiting intercorrelations between marker coordinates.

        See:
            Federolf PA. 2013. PLoS ONE 8(10):e78689. doi:10.1371/journal.pone.0078689
            Gløersen Ø, Federolf P. 2016. PLoS ONE 11(3):e0152616. doi:10.1371/journal.pone.0152616

        Parameters
        ----------
        data_gaps : (N, M) array_like
            Array of marker position data with N time steps across M channels.
            The data need to be organized as follows:

            x1(t1) y1(t1) z1(t1) x2(t1) y2(t1) z2(t1) ...    xm(t1) ym(t1) zm(t1)
            x1(t2) y1(t2) z1(t2) x2(t2) y2(t2) z2(t2) ...    xm(t2) ym(t2) zm(t2)
            ...    ...    ...    ...    ...    ...    ...    ...    ...    ...
            x1(tn) y1(tn) z1(tn) x2(tn) y2(tn) z2(tn) ...    xm(tn) ym(tn) zm(tn)

            Thus, the first three columns correspond to the x-, y-, and-z coordinate of the 1st marker.
            The rows correspond to the consecutive time steps (i.e., frames).
        
        Optional parameters
        -------------------
            method : str
                Reconstruction strategy for gaps in multiple markers (`R1` or `R2` (default)).
            weight_scale : float
                Parameter `sigma` for determining weights. Default is 200.
            mm_weight : float
                Parameter for weight on missing markers. Default is 0.02.
            distal_threshold : float
                Cut-off distance for distal marker in `R2` relative to average Euclidean distances. Default is 0.5.
            min_cum_sv : float
                Minimum cumulative sum of eigenvalues of the normalized singular values.
                Determines the number of principal component vectors included in the analysis. Default is 0.99.
        """
        LOGGER.logger.info("----Gloersen2016 gap filling----")
        btkmarkersLoaded  = btkTools.GetMarkerNames(acq)
        ff = acq.GetFirstFrame()
        lf = acq.GetLastFrame()
        pfn = acq.GetPointFrameNumber()

        if "markers" in kwargs:
            btkmarkers = kwargs["markers"]
        else:
            btkmarkers =[]
            for ml in btkmarkersLoaded:
                if btkTools.isPointExist(acq,ml) :
                    btkmarkers.append(ml)
        # --------

        LOGGER.logger.debug("Populating data matrix")
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

        data_gaps = rawDatabtk


        # Parse optional arguments, or get defaults
        method = kwargs.get("method", "R2")
        weight_scale = kwargs.get("weight_scale", 200)
        mm_weight = kwargs.get("mm_weight", 0.02)
        distal_threshold = kwargs.get("distal_threshold", 0.5)
        min_cum_sv = kwargs.get("min_cum_sv", 0.99)

        # Get number of time steps, number of channels, and corresponding number of markers
        n_time_steps, n_channels = data_gaps.shape
        n_markers = n_channels // 3

        # Find channels with gaps
        ix_channels_with_gaps, = np.nonzero(np.any(np.isnan(data_gaps), axis=0))
            
        # Find time steps with gaps
        ix_time_steps_with_gaps, = np.nonzero(np.any(np.isnan(data_gaps), axis=1))
        
        # If no gaps were found
        if len(ix_time_steps_with_gaps) == 0:
            warnings.warn("Submitted data appear to have no gaps. Make sure that gaps are represented by NaNs.")
            return data_gaps
        elif len(ix_time_steps_with_gaps) == n_time_steps:
            if method == "R1":
                warnings.warn("For each time step there is at least one marker with missing data. Cannot perform reconstruction according to strategy R1.")
                return None
        
        # Subtract mean marker trajectory to get a coordinate system moving with the subject
        T = np.delete(data_gaps, ix_channels_with_gaps, axis=1)
        mean_trajectory_x = np.mean(T[:,::3], axis=1)
        mean_trajectory_y = np.mean(T[:,1::3], axis=1)
        mean_trajectory_z = np.mean(T[:,2::3], axis=1)
        del T

        B = data_gaps.copy()
        B[:,::3] = B[:,::3] - np.tile(mean_trajectory_x.reshape(-1,1), (1, n_markers))
        B[:,1::3] = B[:,1::3] - np.tile(mean_trajectory_y.reshape(-1,1), (1, n_markers))
        B[:,2::3] = B[:,2::3] - np.tile(mean_trajectory_z.reshape(-1,1), (1, n_markers))

        # Reconstruct missing marker data
        if method == "R1":
            reconstructed_data = self._reconstruct(B, weight_scale=weight_scale, mm_weight=mm_weight, min_cum_sv=min_cum_sv)

            # Replace the missing data with reconstructed data
            filled_data = np.where(np.isnan(data_gaps), reconstructed_data, B)
        elif method == "R2":
            # Allocate space for reconstructed data matrix
            reconstructed_data = B.copy()

            # Get markers with gaps
            ix_markers_with_gaps = ix_channels_with_gaps[2::3] // 3
            for ix in ix_markers_with_gaps:
                eucl_distance_2_markers = self._distance2marker(B, np.arange(ix*3,(ix+1)*3))
                thresh = distal_threshold * np.mean(eucl_distance_2_markers)
                ix_channels_2_zero = np.argwhere(np.logical_and(np.reshape(np.tile(eucl_distance_2_markers, (3,1)), (1,n_channels), order="F").reshape(-1,) > thresh, \
                    np.any(np.isnan(B), axis=0)))[:,0]
                
                # Set channels to 0, for which there are NaNs and that are far away from the current marker
                data_gaps_removed_cols = B.copy()
                data_gaps_removed_cols[:,ix_channels_2_zero] = 0
                data_gaps_removed_cols[:,ix*3:(ix+1)*3] = B[:,ix*3:(ix+1)*3]

                # Find gaps in marker trajectory
                ix_frames_with_gaps, = np.nonzero(np.isnan(B[:, ix]))
                
                # For channels that have gaps in the same time span, set values to 0
                for jx in np.setdiff1d(ix_markers_with_gaps, ix):
                    if np.any(np.isnan(data_gaps_removed_cols[ix_frames_with_gaps,3*jx])):
                        data_gaps_removed_cols[:,3*jx:3*jx+3] = 0
                
                # Find frames without gaps in marker trajectory
                ix_frames_no_gaps, = np.nonzero(np.logical_not(np.any(np.isnan(data_gaps_removed_cols), axis=1)))

                # Find frames with gaps in marker trajectory `ix`
                ix_frames_2_reconstruct, = np.nonzero(np.any(np.isnan(data_gaps_removed_cols[:,3*ix:3*ix+3]), axis=1))

                # Concatenate frames to reconstruct, at the end frames without gaps
                ix_complete_and_gapped_frames = np.concatenate((ix_frames_no_gaps, ix_frames_2_reconstruct))

                # Get indexes of frames to fill
                ix_fill_frames = np.arange(len(ix_frames_no_gaps), len(ix_complete_and_gapped_frames))

                # Store temporarily reconstruct data
                temp_reconstructed_data = self._reconstruct(data_gaps_removed_cols[ix_complete_and_gapped_frames,:], weight_scale=weight_scale, mm_weight=mm_weight, min_cum_sv=min_cum_sv)

                # Replace gapped data with reconstructed data
                reconstructed_data[ix_frames_2_reconstruct, 3*ix:3*ix+3] = temp_reconstructed_data[ix_fill_frames, 3*ix:3*ix+3]
            
            # Assign to output variable
            filled_data = reconstructed_data.copy()
        else:
            warnings.warn("Invalid reconstruction method, please specify `R1` or `R2`. Returning original data.")
            return data_gaps

        # Add the mean marker trajectory
        filled_data[:,::3] = filled_data[:,::3] + np.tile(mean_trajectory_x.reshape(-1,1), (1,n_markers))
        filled_data[:,1::3] = filled_data[:,1::3] + np.tile(mean_trajectory_y.reshape(-1,1), (1,n_markers))
        filled_data[:,2::3] = filled_data[:,2::3] + np.tile(mean_trajectory_z.reshape(-1,1), (1,n_markers))
        
        
        # Create new smoothed trjectories
        filledMarkers  = list()
        for i in range(0,len(btkmarkers)):
            targetMarker = btkmarkers[i]
            if btkTools.isGap(acq,targetMarker):
                LOGGER.logger.info("marker (%s) --> filled"%(targetMarker))
                filledMarkers.append(targetMarker)
                val_final = np.zeros((pfn,3))
                val_final[:,0] = filled_data[:,3*i-3]
                val_final[:,1] = filled_data[:,3*i-2]
                val_final[:,2] = filled_data[:,3*i-1]
                btkTools.smartAppendPoint(acq,targetMarker,val_final)
        LOGGER.logger.info("----Gloersen2016 gap filling [complete]----")
        
        
        
        
        return acq, filledMarkers