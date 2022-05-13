# from __future__ import division
# import numpy as np
# from . import NexusObject


class NexusAngles(NexusObject):

    def __init__(self, subject_name=None):
        super(NexusAngles, self).__init__(subject_name)
        self._angles = np.array([np.nan, np.nan, np.nan], dtype=np.float32, ndmin=2)

    def Create(self, name, SDK):

        if name not in SDK.GetModelOutputNames(self.subject_name):
            ComponentNames = ['X', 'Y', 'Z']
            Types = ['Angle', 'Angle', 'Angle']
            SDK.CreateModelOutput(self.subject_name, name, 'Angles', ComponentNames, Types)

    def Read(self, name, SDK):

        if name in SDK.GetModelOutputNames(self.subject_name):
            angles, e = SDK.GetModelOutput(self.subject_name, name)
            angles = np.array(angles, dtype=np.float32, ndmin=2)
        else:
            raise ValueError('No model output named {}'.format(name))

        angles[~e, :] = np.nan
        self._angles = angles.T

    def Write(self, name, SDK):

        angles = np.array(self._angles, dtype=float)
        exists = np.all(np.isfinite(angles), axis=1)
        angles[~exists, :] = 0

        frameCount = SDK.GetFrameCount()
        frames = np.shape(self._angles)[0]
        if frames == 1:
            angles = np.resize(angles, (frameCount, 3))
            exists = np.resize(exists, frameCount)

        elif frames < frameCount:
            angles = np.resize(angles, (frameCount, 3))
            angles[frames + 1:, :] = 0

            exists = np.resize(exists, frameCount)
            exists[frames + 1:] = False

        elif frames == frameCount:
            pass

        else:
            raise ValueError('Too many output points for trial')

        SDK.SetModelOutput(self.subject_name, name, angles.T, exists)

        # TODO: get angles as different representations

    # store as angle-axis, allow users to return this as euler, fixed...
    def Angles(self, frames):

        if frames is None:
            return self._angles.tolist()

        if np.isscalar(frames):
            return self._angles[frames - 1]

        frameIndices = np.array(frames - 1, dtype=np.long)
        return self._angles[frameIndices].tolist()

    # TODO: allow user to specify what representation they have used (euler, fixed...)
    # convert to angle-axis for storage
    def SetAngles(self, components, frames=None):

        component_array = np.array(components, dtype=np.float32, ndmin=2)

        if not np.shape(component_array)[1] == 3:
            raise TypeError('components must be an Nx3 array')

        if frames is None:
            self._angles = component_array
            return

        frameIndices = np.array(frames - 1, dtype=np.long)
        if np.shape(frameIndices)[0] != np.shape(components)[0]:
            raise ValueError('Number of positions must match number of frames')

        last_frame = max(frames)
        num_positions = len(self._angles)

        for i, index in enumerate(frameIndices):
            self._angles[index] = component_array[i]

        for i in range(last_frame, num_positions):
            np.append(self._angles, [[np.nan, np.nan, np.nan]], axis=0)

    def __pos__(self):
        result = NexusAngles(self.subject_name)
        result._angles = self._angles
        return result

    def __neg__(self):
        result = NexusAngles(self.subject_name)
        result._angles = -self._angles
        return result

    def __mul__(self, other):
        # This will be true once we start only storing angle-axes
        # '''Multiply the magnitude of the rotation. Does not affect the axis of rotation'''
        if not np.isscalar(other):
            return NotImplemented

        result = NexusAngles(self.subject_name)
        result._angles = self._angles * other
        return result

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        # This will be true once we start only storing angle-axes
        # '''Divide the magnitude of the rotation. Does not affect the axis of rotation'''
        if not np.isscalar(other):
            return NotImplemented

        result = NexusAngles(self.subject_name)
        result._angles = self._angles / other
        return result

    def mean(self):
        result = NexusAngles(self.subject_name)
        result._position = np.mean(self._angles, axis=0)
        return result
