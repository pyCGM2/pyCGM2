# from __future__ import division
# import numpy as np
# from viconnexusapi import ViconNexus
# from . import NexusObject
# try:
#     from . import NexusSegment
# except ImportError:
#     import NexusSegment


class NexusTrajectory(NexusObject):

    def __init__(self, subject_name=None):

        super(NexusTrajectory, self).__init__(subject_name)
        self._position = np.array([np.nan, np.nan, np.nan], dtype=np.float32, ndmin=2)

    def Create(self, name, SDK):

        if not isinstance(SDK, ViconNexus.ViconNexus):
            raise TypeError('SDK object must be of type ViconNexus')

        if name not in SDK.GetModelOutputNames(self.subject_name):
            SDK.CreateModeledMarker(self.subject_name, name)

    def Read(self, name, SDK):

        if not isinstance(SDK, ViconNexus.ViconNexus):
            raise TypeError('SDK object must be of type ViconNexus')

        subject = self.subject_name
        if name in SDK.GetMarkerNames(subject):
            x, y, z, e = SDK.GetTrajectory(subject, name)
            position = np.array([x, y, z], dtype=np.float32, ndmin=2)

        elif name in SDK.GetModelOutputNames(subject):
            position, e = SDK.GetModelOutput(subject, name)
            position = np.array(position, dtype=np.float32, ndmin=2)

        else:
            raise ValueError('No model output or trajectory named {}'.format(name))

        e = np.array(e, dtype=np.bool)
        position[:, ~e] = np.nan

        self._position = position.T

    def Write(self, name, SDK):

        if not isinstance(SDK, ViconNexus.ViconNexus):
            raise TypeError('SDK object must be of type ViconNexus')

        position = np.array(self._position, dtype=float)
        exists = np.all(np.isfinite(position), axis=1)
        position[~exists, :] = 0

        frameCount = SDK.GetFrameCount()
        frames = np.shape(self._position)[0]
        if frames == 1:
            position = np.resize(position, (frameCount, 3))
            exists = np.resize(exists, frameCount)

        elif frames < frameCount:
            position = np.resize(position, (frameCount, 3))
            position[frames + 1:, :] = 0

            exists = np.resize(exists, frameCount)
            exists[frames + 1:] = False

        elif frames == frameCount:
            pass

        else:
            raise ValueError('Too many output points for trial')

        SDK.SetModelOutput(self.subject_name, name, position.T, exists)

    def Position(self, frames=None):

        if frames is None:
            return self._position.tolist()

        if np.isscalar(frames):
            return self._position[frames - 1]

        frameIndices = np.array(frames - 1, dtype=np.long)
        return self._position[frameIndices].tolist()

    def SetPosition(self, components, frames=None):

        component_array = np.array(components, dtype=np.float32, ndmin=2)
        if not np.shape(component_array)[1] == 3:
            raise TypeError('components must be an Nx3 array')

        if frames is None:
            self._position = component_array
            return

        frameIndices = np.array(frames - 1, dtype=np.long)
        if np.shape(frameIndices)[0] != np.shape(components)[0]:
            raise ValueError('Number of positions must match number of frames')

        last_frame = max(frames)
        num_positions = len(self._position)

        for i, index in enumerate(frameIndices):
            self._position[index] = component_array[i]

        for i in range(last_frame, num_positions):
            np.append(self._position, [[np.nan, np.nan, np.nan]], axis=0)

    def __radd__(self, other):
        return self + other

    def __add__(self, other):

        result = NexusTrajectory()
        subject_b = ''

        if isinstance(other, NexusTrajectory):
            position_b = other._position
            subject_b = other._subject_name
        else:
            try:
                position_b = np.array(other, dtype=np.float32, ndmin=2)
            except Exception:
                return NotImplemented

        if self.subject_name and subject_b:
            if self.subject_name == subject_b:
                result.subject_name = self.subject_name
        elif self.subject_name:
            result.subject_name = self.subject_name
        elif subject_b:
            result.subject_name = subject_b

        result.SetPosition(self._position + position_b)

        return result

    def __sub__(self, other):

        result = NexusTrajectory()

        subject_b = ''

        if isinstance(other, NexusTrajectory):
            position_b = other._position
            subject_b = other.subject_name
        else:
            try:
                position_b = np.array(other, dtype=np.float32, ndmin=2)
            except Exception:
                return NotImplemented

        if self.subject_name and subject_b:
            if self.subject_name == subject_b:
                result.subject_name = self.subject_name
        elif self.subject_name:
            result.subject_name = self.subject_name
        elif subject_b:
            result.subject_name = subject_b

        result._position = self._position - position_b

        return result

    def __mul__(self, other):
        if isinstance(other, NexusSegment.NexusSegment):
            subject = NexusObject.NexusObject._get_subject_name(self, other)
            result = NexusTrajectory(subject)
            result.SetPosition(other.GlobalisePoint(self._position))
            return result

        if not np.isscalar(other):
            return NotImplemented

        result = NexusTrajectory(self.subject_name)
        result._position = self._position * other
        return result

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        if isinstance(other, NexusSegment.NexusSegment):
            subject = NexusObject.NexusObject._get_subject_name(self, other)
            result = NexusTrajectory(subject)
            result.SetPosition(other.LocalisePoint(self._position))
            return result

        if not np.isscalar(other):
            return NotImplemented

        result = NexusTrajectory(self.subject_name)
        result._position = self._position / other
        return result

    def __neg__(self):
        result = NexusTrajectory(self.subject_name)
        result._position = -self._position
        return result

    def __pos__(self):
        result = NexusTrajectory(self.subject_name)
        result._position = self._position
        return result

    def mean(self):
        result = NexusTrajectory(self.subject_name)
        result._position = np.mean(self._position, axis=0)
        return result
NexusObject
