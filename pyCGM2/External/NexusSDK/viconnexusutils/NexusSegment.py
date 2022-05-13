# from __future__ import division
# from six import string_types
# import numpy as np
#
# from viconnexusapi import ViconUtils
# from . import NexusAngles
# from . import NexusObject
# from . import NexusTrajectory


class NexusSegment(NexusObject):

    def __init__(self, subject=None, axisOrder='xyz'):

        super(NexusSegment, self).__init__(subject)
        self.axisOrder = axisOrder
        self._translation = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32, ndmin=2)
        self._orientation = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32, ndmin=2)
        self._scale = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32, ndmin=2)

    @staticmethod
    def global_segment():

        Segment = NexusSegment()
        Segment._translation = np.array([[0, 0, 0]], dtype=np.float32, ndmin=2)
        Segment._orientation = np.array([[0, 0, 0]], dtype=np.float32, ndmin=2)
        Segment._scale = np.array([[1, 1, 1]], dtype=np.float32, ndmin=2)
        return Segment

    @property
    def axisOrder(self):

        return self._axisOrder

    @axisOrder.setter
    def axisOrder(self, axisOrder):

        if not isinstance(axisOrder, string_types) or \
                'x' not in axisOrder or \
                'y' not in axisOrder or \
                'z' not in axisOrder:
            raise ValueError('Invalid axis order. Specify a combination of ''xyz''')

        self._axisOrder = axisOrder

    def Create(self, name, SDK):

        if name not in SDK.GetModelOutputNames(self.subject_name):
            ComponentNames = ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'SX', 'SY', 'SZ']
            Types = ['Angle', 'Angle', 'Angle', 'Length', 'Length', 'Length', 'Length', 'Length', 'Length']
            SDK.CreateModelOutput(self.subject_name, name, 'Plug-in Gait Bones', ComponentNames, Types)

    def Read(self, name, SDK):

        if name in SDK.GetModelOutputNames(self.subject_name):

            Data, e = SDK.GetModelOutput(self.subject_name, name)
            Data = np.array(Data, dtype=np.float32, ndmin=2)
            e = np.array(e, dtype=np.bool)
            Data[:, ~e] = np.nan

        else:
            raise ValueError('No model output named {}'.format(name))

        AngleAxis = Data[0:3, :] * np.pi / 180

        def convert_angle_axis(rotation):
            matrix = ViconUtils.RotationMatrixFromAngleAxis(rotation.tolist())
            return ViconUtils.EulerFromMatrix(matrix, self.axisOrder)

        self._orientation = np.array(list(map(convert_angle_axis, AngleAxis.T)), dtype=np.float32, ndmin=2)
        self._translation = Data[3:6, :].T
        self._scale = Data[6:9, :].T

    def Write(self, name, SDK):

        def convert_rotation(rotation):
            matrix = ViconUtils.MatrixFromEuler(rotation, self.axisOrder)
            return ViconUtils.AngleAxisFromMatrix(matrix)

        data = list(map(convert_rotation, self._orientation.tolist()))
        data = np.array(data, dtype=float)
        data *= 180 / np.pi
        data = np.append(data, self._translation, axis=1)
        data = np.append(data, self._scale, axis=1)

        exists = np.all(np.isfinite(data), axis=1)
        data[~exists, :] = 0

        frameCount = SDK.GetFrameCount()
        frames = np.shape(data)[0]
        if frames == 1:
            data = np.resize(data, (frameCount, 9))
            exists = np.resize(exists, frameCount)

        elif frames < frameCount:
            data = np.resize(data, (frameCount, 9))
            data[frames + 1:, :] = 0

            exists = np.resize(exists, frameCount)
            exists[frames + 1:] = False

        elif frames == frameCount:
            pass

        else:
            raise ValueError('Too many output points for trial')

        SDK.SetModelOutput(self.subject_name, name, data.T, exists)

        # TODO: Nexus only cares about the scale of the Z-axis

    # we currently use the scale of the primary axis here...
    def Populate(self, origin, firstAxis, crossingAxis, axisOrder=None):
        """Define a segment from three trajectories
        The first defines the location of the origin
        The second is the direction of the primary axis
        The third defines the plane of the second axis by b x a
        The order of the axes may optionally be specified (default xyz)"""

        bSubjectOk = True
        subject = self.subject_name

        if axisOrder is None:
            if self.axisOrder:
                axisOrder = self.axisOrder
            else:
                axisOrder = 'xyz'

        if isinstance(origin, NexusTrajectory.NexusTrajectory):
            translation = np.array(origin.Position(), dtype=np.float32, ndmin=2)
            if not subject and origin.subject_name:
                subject = origin.subject_name
            else:
                bSubjectOk = False
        else:
            translation = np.array(origin, dtype=np.float32, ndmin=2)

        if np.shape(translation)[1] != 3:
            raise TypeError('Invalid segment origin: must be a NexusTrajectory or a Nx3 array')

        if isinstance(firstAxis, NexusTrajectory.NexusTrajectory):
            axisA = np.array(firstAxis.Position(), dtype=np.float32, ndmin=2)
            if subject != firstAxis.subject_name:
                if not subject:
                    subject = firstAxis.subject_name
                else:
                    bSubjectOk = False
        else:
            axisA = np.array(firstAxis, dtype=np.float32, ndmin=2)

        if np.shape(axisA)[1] != 3:
            raise TypeError('Invalid primary axis: must be a NexusTrajectory or a Nx3 array''')

        if isinstance(crossingAxis, NexusTrajectory.NexusTrajectory):
            axisB = np.array(crossingAxis.Position(), dtype=np.float32, ndmin=2)
            if subject != crossingAxis.subject_name:
                if not subject:
                    subject = crossingAxis.subject_name
                else:
                    bSubjectOk = False
        else:
            axisB = np.array(crossingAxis, dtype=np.float32, ndmin=2)

        if np.shape(axisB)[1] != 3:
            raise TypeError('Invalid crossing axis: must be a NexusTrajectory or a Nx3 array''')

        if np.shape(translation) != np.shape(axisA) or np.shape(axisA) != np.shape(axisB):
            raise ValueError('Mismatched sizes: all inputs must be the same length')

        scaleA = np.linalg.norm(axisA, axis=1, keepdims=True)
        scaleB = np.linalg.norm(axisB, axis=1, keepdims=True)

        axis1 = axisA / scaleA
        axis2 = np.cross(axisB / scaleB, axis1)
        axis2 = axis2 / np.linalg.norm(axis2, axis=1, keepdims=True)
        axis3 = np.cross(axis1, axis2)
        axis3 = axis3 / np.linalg.norm(axis3, axis=1, keepdims=True)

        if axisOrder in ['xzy', 'yxz', 'zyx']:
            axis3 = -axis3

        orientation = np.array(np.nan, dtype=np.float32, ndmin=2)
        orientation = np.resize(orientation, np.shape(translation))
        for i in range(np.shape(translation)[0]):
            col1 = eval('axis{}[i]'.format(axisOrder.find('x') + 1))
            col2 = eval('axis{}[i]'.format(axisOrder.find('y') + 1))
            col3 = eval('axis{}[i]'.format(axisOrder.find('z') + 1))

            RotMat = list(map(list, zip(*[col1, col2, col3])))
            orientation[i] = ViconUtils.EulerFromMatrix(RotMat, axisOrder)

        if bSubjectOk:
            self.subject_name = subject
        self._scale = np.resize(np.array(scaleA, dtype=np.float32), np.shape(translation.T)).T
        self._orientation = orientation
        self._translation = translation
        self.axisOrder = axisOrder

    def Position(self, frames=None):
        """
        Get the position of the segment origin.
        If ''frames'' is supplied, only positions at the requested frames are returned
        """

        if frames is None:
            return self._translation.tolist()

        if np.isscalar(frames):
            return self._translation[frames - 1]

        frameIndices = np.array(frames - 1, dtype=np.long)
        return self._translation[frameIndices].tolist()

    def SetPosition(self, components, frames=None):
        """Sets the position of the segment origin. Can optionally update at specified frame numbers"""

        component_array = np.array(components, dtype=np.float32, ndmin=2)

        if np.shape(component_array)[1] != 3:
            raise ValueError('Parameter ''components'' must be a n x 3 array')

        if frames is None:
            self._translation = component_array
            return

        if np.isscalar(frames):
            self._translation[frames - 1] = component_array

        frameIndices = np.array(frames, dtype=np.long)
        if np.shape(frameIndices)[0] != np.shape(component_array)[0]:
            raise ValueError('Number of positions must match number of frames')

        last_frame = max(frames)
        num_frames = len(self._translation)

        for i, index in enumerate(frameIndices):
            self._translation[index] = component_array[i]

        for i in range(last_frame, num_frames):
            np.append(self._translation, [[np.nan, np.nan, np.nan]], axis=0)

    def Orientation(self, frames=None):
        """
        Get the orientation of the segment.
        If ''frames'' is supplied, only orientations at the requested frames are returned
        """

        if frames is None:
            return self._orientation.tolist()

        if np.isscalar(frames):
            return self._orientation[frames - 1]

        frameIndices = np.array(frames - 1, dtype=np.long)
        return self._orientation[frameIndices].tolist()

    def SetOrientation(self, components, frames=None):
        """Sets the orientation of the segment. Can optionally update at specified frame numbers"""

        component_array = np.array(components, dtype=np.float32, ndmin=2)

        if np.shape(component_array)[1] != 3:
            raise ValueError('Parameter ''components'' must be a n x 3 array')

        if frames is None:
            self._orientation = component_array
            return

        if np.isscalar(frames):
            self._orientation[frames - 1] = component_array

        frameIndices = np.array(frames, dtype=np.long)
        if np.shape(frameIndices)[0] != np.shape(component_array)[0]:
            raise ValueError('Number of positions must match number of frames')

        last_frame = max(frames)
        num_frames = len(self._orientation)

        for i, index in enumerate(frameIndices):
            self._orientation[index] = component_array[i]

        for i in range(last_frame, num_frames):
            np.append(self._orientation, [[np.nan, np.nan, np.nan]], axis=0)

    def Scale(self, frames=None):
        """Get the scale of the segment. If ''frames'' is supplied, only scales at the requested frames are returned"""

        if frames is None:
            return self._orientation.tolist()

        if np.isscalar(frames):
            return self._orientation[frames - 1]

        frameIndices = np.array(frames - 1, dtype=np.long)
        return self._orientation[frameIndices].tolist()

    def SetScale(self, components, frames=None):
        """Sets the scale of the segment. Can optionally update at specified frame numbers"""

        component_array = np.array(components, dtype=np.float32, ndmin=2)

        if np.shape(component_array)[1] != 3:
            raise ValueError('Parameter ''components'' must be a n x 3 array')

        if frames is None:
            self._scale = component_array
            return

        if np.isscalar(frames):
            self._scale[frames - 1] = component_array

        frameIndices = np.array(frames, dtype=np.long)
        if np.shape(frameIndices)[0] != np.shape(component_array)[0]:
            raise ValueError('Number of positions must match number of frames')

        last_frame = max(frames)
        num_frames = len(self._scale)

        for i, index in enumerate(frameIndices):
            self._scale[index] = component_array[i]

        for i in range(last_frame, num_frames):
            np.append(self._scale, [[np.nan, np.nan, np.nan]], axis=0)

    def GlobalisePoint(self, point, frames=None):
        """Return the global position of the specified segment-local point """

        if frames is None:
            frameIndices = np.array(range(len(self._translation)), dtype=np.long)
        else:
            frameIndices = np.array(frames - 1, dtype=np.long)

        globalPoint = np.resize(np.array(np.nan, dtype=np.float32, ndmin=2), (np.shape(frameIndices)[0], 3))

        if isinstance(point, NexusTrajectory.NexusTrajectory):
            position = np.array(point.Position(frames), dtype=np.float32, ndmin=2)
        else:
            position = np.array(point, dtype=np.float32, ndmin=2)

        if np.shape(position)[0] == 1:
            for i, frame in enumerate(frameIndices):
                poseR = ViconUtils.MatrixFromEuler(self._orientation[frame], self.axisOrder)
                poseR = np.array(poseR).flatten()
                globalPoint[i] = ViconUtils.Globalise(position[0], poseR, self._translation[frame])
        else:
            for i, frame in enumerate(frameIndices):
                poseR = ViconUtils.MatrixFromEuler(self._orientation[frame], self.axisOrder)
                poseR = np.array(poseR).flatten()
                globalPoint[i] = ViconUtils.Globalise(position[i], poseR, self._translation[frame])

        return globalPoint.tolist()

    def LocalisePoint(self, point, frames=None):
        """Return the segment-local position of the specified global point """

        if frames is None:
            frameIndices = np.array(range(len(self._translation)), dtype=np.long)
        else:
            frameIndices = np.array(frames - 1, dtype=np.long)

        localPoint = np.resize(np.array(np.nan, dtype=np.float32, ndmin=2), (np.shape(frameIndices)[0], 3))

        if isinstance(point, NexusTrajectory.NexusTrajectory):
            position = np.array(point.Position(frames), dtype=np.float32, ndmin=2)
        else:
            position = np.array(point, dtype=np.float32, ndmin=2)

        if np.shape(position)[0] == 1:
            for i, frame in enumerate(frameIndices):
                poseR = ViconUtils.MatrixFromEuler(self._orientation[frame], self.axisOrder)
                poseR = np.array(poseR).flatten()
                localPoint[i] = ViconUtils.Localise(position[0], poseR, self._translation[frame])
        else:
            for i, frame in enumerate(frameIndices):
                poseR = ViconUtils.MatrixFromEuler(self._orientation[frame], self.axisOrder)
                poseR = np.array(poseR).flatten()
                localPoint[i] = ViconUtils.Localise(position[i], poseR, self._translation[frame])

        return localPoint.tolist()

    def TranslatePointInSegment(self, point, translation, frames=None):
        """
        Returns a NexusTrajectory whose position is ''point'' translated by ''translation'' in the segment local
        coordinate system
        """

        if frames is None:
            frameIndices = np.array(range(len(self._translation)), dtype=np.long)
        else:
            frameIndices = np.array(frames - 1, dtype=np.long)

        translatedPoint = np.resize(np.array(np.nan, dtype=np.float32, ndmin=2), (np.shape(frameIndices)[0], 3))

        subject = ''
        if isinstance(point, NexusTrajectory.NexusTrajectory):
            position = np.array(point.Position(frames), dtype=np.float32, ndmin=2)
            subject = point.subject_name
        else:
            position = np.array(point, dtype=np.float32, ndmin=2)

        if np.shape(position)[0] == 1:
            for i, frame in enumerate(frameIndices):
                poseR = ViconUtils.MatrixFromEuler(self._orientation[frame], self.axisOrder)
                poseR = np.array(poseR).flatten()
                localPoint = np.array(ViconUtils.Localise(position, poseR, self._translation[frame])) + translation
                translatedPoint[i] = ViconUtils.Globalise(localPoint, poseR, self._translation[frame])
        else:
            for i, frame in enumerate(frameIndices):
                poseR = ViconUtils.MatrixFromEuler(self._orientation[frame], self.axisOrder)
                poseR = np.array(poseR).flatten()
                localPoint = np.array(ViconUtils.Localise(position[i], poseR, self._translation[frame])) + translation
                translatedPoint[i] = ViconUtils.Globalise(localPoint, poseR, self._translation[frame])

        result = NexusTrajectory.NexusTrajectory(subject)
        result.SetPosition(translatedPoint, frames)
        return result

        # TODO: this should only return helical angle, once that's what NexusAngles expects?

    # NexusAngles can then take care of transforming the result into whatever the user wants
    def AngleBetween(self, segment, angle_type='euler', order='xyz'):
        """Return the angle between this segment and the one specified"""

        if not isinstance(segment, NexusSegment):
            raise TypeError('NexusSegments can only be compared against other NexusSegments')

        if angle_type not in ['fixed', 'euler', 'helical']:
            raise ValueError('Invalid angle type: choose ''fixed'', ''euler'' or ''helical''')

        rot1 = self._orientation
        rot2 = segment._orientation

        if np.shape(rot1)[0] == 1:
            rot1 = np.resize(rot1, np.shape(rot2))

        if np.shape(rot2)[0] == 1:
            rot2 = np.resize(rot2, np.shape(rot1))

        if np.shape(rot1) != np.shape(rot2):
            raise ValueError('Incompatible segments: frame counts must be the same')

        def ComputeAngle(x, y):
            rotA = np.array(ViconUtils.MatrixFromEuler(x, self.axisOrder), ndmin=2)
            rotB = np.array(ViconUtils.MatrixFromEuler(y, segment.axisOrder), ndmin=2)
            matrix = np.matmul(rotA.T, rotB)

            angle = np.array([np.nan, np.nan, np.nan])
            if angle_type == 'helical':
                angle = np.array(ViconUtils.AngleAxisFromMatrix(matrix))

            elif angle_type == 'euler':
                euler_angles = ViconUtils.EulerFromMatrix(matrix, order)
                angle[order.find('x')] = euler_angles[0]
                angle[order.find('y')] = euler_angles[1]
                angle[order.find('z')] = euler_angles[2]
                angle *= 180 / np.pi

            elif angle_type == 'fixed':
                euler_angles = ViconUtils.EulerFromMatrix(matrix.T, order)
                angle[order.find('x')] = -euler_angles[0]
                angle[order.find('y')] = -euler_angles[1]
                angle[order.find('z')] = -euler_angles[2]
                angle *= 180 / np.pi

            return angle.tolist()

        angle_output = list(map(ComputeAngle, rot1, rot2))

        result = NexusAngles.NexusAngles()
        subject = NexusObject.NexusObject._get_subject_name(self, segment)
        result.subject_name = subject
        result.SetAngles(angle_output)
        return result

    def __add__(self, trajectory):
        """Move a segment origin by a trajectory"""

        if not isinstance(trajectory, NexusTrajectory.NexusTrajectory):
            raise TypeError('Only a NexusTrajectory can be added to a NexusSegment')

        subject = NexusObject.NexusObject._get_subject_name(self, trajectory)
        result = NexusSegment(subject, self.axisOrder)
        result._translation = self._translation + trajectory.Position()
        result._orientation = self._orientation
        result._scale = self._scale
        return result

    def __radd__(self, trajectory):

        return self + trajectory

    def __sub__(self, trajectory):

        if not isinstance(trajectory, NexusTrajectory.NexusTrajectory):
            raise TypeError('Only a NexusTrajectory can be subtracted from a NexusSegment')

        subject = NexusObject.NexusObject._get_subject_name(self, trajectory)
        result = NexusSegment(subject, self.axisOrder)
        result._translation = self._translation - trajectory.Position()
        result._orientation = self._orientation
        result._scale = self._scale
        return result

    def __rmul__(self, trajectory):
        pass

    def mean(self):
        result = NexusSegment(self.subject_name, self.axisOrder)
        result._translation = np.mean(self._translation, axis=0)
        result._orientation = np.mean(self._orientation, axis=0)  # does this make sense? probably not
        result._scale = np.mean(self._scale, axis=0)
        pass
