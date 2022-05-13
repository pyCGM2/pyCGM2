# """
# module docstring
# A collection of utilities
# """
#
# import math
# import sys


def CreateViconGaitModelOutputs(ViconNexus):
    # create the model outputs formatted to match Vicon Plug-In_Gait
    # model outputs will be created for each subject in the workspace

    # get a list of the currently loaded subjects
    subjects = ViconNexus.GetSubjectNames()

    for subjectname in subjects:
        XYZNames = ['X', 'Y', 'Z']

        # Angles
        AnglesTypes = ['Angle', 'Angle', 'Angle']
        ViconNexus.CreateModelOutput(subjectname, 'LAbsAnkleAngle', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LAnkleAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LElbowAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LFootProgressAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LHeadAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LHipAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LKneeAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LNeckAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LPelvisAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LShoulderAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LSpineAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LThoraxAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LWristAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RAbsAnkleAngle', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RAnkleAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RElbowAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RFootProgressAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RHeadAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RHipAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RKneeAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RNeckAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RPelvisAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RShoulderAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RSpineAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RThoraxAngles', 'Angles', XYZNames, AnglesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RWristAngles', 'Angles', XYZNames, AnglesTypes)

        # Forces
        ForcesTypes = ['Force', 'Force', 'Force']
        ForcesNormalizedTypes = ['ForceNormalized', 'ForceNormalized', 'ForceNormalized']
        ViconNexus.CreateModelOutput(subjectname, 'LAnkleForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LElbowForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LGroundReactionForce', 'Forces', XYZNames, ForcesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LHipForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LKneeForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LNeckForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LNormalisedGRF', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LShoulderForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LWaistForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LWristForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RAnkleForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RElbowForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RGroundReactionForce', 'Forces', XYZNames, ForcesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RHipForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RKneeForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RNeckForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RNormalisedGRF', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RShoulderForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RWaistForce', 'Forces', XYZNames, ForcesNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RWristForce', 'Forces', XYZNames, ForcesNormalizedTypes)

        # Moments
        MomentsTypes = ['Torque', 'Torque', 'Torque']
        MomentsNormalizedTypes = ['TorqueNormalized', 'TorqueNormalized', 'TorqueNormalized']
        ViconNexus.CreateModelOutput(subjectname, 'LAnkleMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LElbowMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LGroundReactionMoment', 'Moments', XYZNames, MomentsTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LHipMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LKneeMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LNeckMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LShoulderMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LWaistMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LWristMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RAnkleMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RElbowMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RGroundReactionMoment', 'Moments', XYZNames, MomentsTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RHipMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RKneeMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RNeckMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RShoulderMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RWaistMoment', 'Moments', XYZNames, MomentsNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RWristMoment', 'Moments', XYZNames, MomentsNormalizedTypes)

        # Plug-in Gait Bones
        BonesNames = ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'SX', 'SY', 'SZ']
        BonesTypes = ['Angle', 'Angle', 'Angle', 'Length', 'Length', 'Length', 'Length', 'Length', 'Length']
        ViconNexus.CreateModelOutput(subjectname, 'HED', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LCL', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LFE', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LFO', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LHN', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LHU', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LRA', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LTI', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LTO', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'PEL', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RCL', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RFE', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RFO', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RHN', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RHU', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RRA', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RTI', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RTO', 'Plug-in Gait Bones', BonesNames, BonesTypes)
        ViconNexus.CreateModelOutput(subjectname, 'TRX', 'Plug-in Gait Bones', BonesNames, BonesTypes)

        # Powers
        PowersNormalizedTypes = ['PowerNormalized', 'PowerNormalized', 'PowerNormalized']
        ViconNexus.CreateModelOutput(subjectname, 'LAnklePower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LElbowPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LHipPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LKneePower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LNeckPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LShoulderPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LWaistPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'LWristPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RAnklePower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RElbowPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RHipPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RKneePower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RNeckPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RShoulderPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RWaistPower', 'Powers', XYZNames, PowersNormalizedTypes)
        ViconNexus.CreateModelOutput(subjectname, 'RWristPower', 'Powers', XYZNames, PowersNormalizedTypes)


def Globalise(point, WorldPoseR, WorldPoseT):
    # given a world pose, globalize the point
    # WorldPoseR - World rotation matrix (row major format, 9 elements)
    # WorldPoseT - World translation in mm (3 elements)

    RotMat = [[0] * 3 for x in range(3)]
    idx = 0
    for i in range(3):
        for j in range(3):
            RotMat[i][j] = WorldPoseR[idx]
            idx = idx + 1

    inputData = point
    TVector = WorldPoseT
    globalisedPoint = [0] * 3
    for i in range(3):
        for j in range(3):
            globalisedPoint[i] = globalisedPoint[i] + (RotMat[i][j] * inputData[j])
    for i in range(3):
        globalisedPoint[i] = globalisedPoint[i] + TVector[i]

    return globalisedPoint


def Localise(point, WorldPoseR, WorldPoseT):
    # given a world pose, localize the point
    # WorldPoseR - World rotation matrix (row major format, 9 elements)
    # WorldPoseT - World translation in mm (3 elements)

    RotMat = [[0] * 3 for x in range(3)]
    idx = 0
    for i in range(3):
        for j in range(3):
            RotMat[i][j] = WorldPoseR[idx]
            idx = idx + 1

    inputData = point
    TVector = WorldPoseT

    # negate matrix
    NegRotMat = [[0] * 3 for x in range(3)]
    for i in range(3):
        for j in range(3):
            NegRotMat[i][j] = RotMat[i][j] * -1

    temp = [0] * 3
    for i in range(3):
        for j in range(3):
            temp[i] = temp[i] + (NegRotMat[j][i] * TVector[j])
    TVector = temp

    # localise the point
    localisedPoint = [0] * 3
    for i in range(3):
        for j in range(3):
            localisedPoint[i] = localisedPoint[i] + (RotMat[j][i] * inputData[j])
    for i in range(3):
        localisedPoint[i] = localisedPoint[i] + TVector[i]

    return localisedPoint


def _safe_arcsin(value):
    """Inverse sin function allowing for floating point precision of inputs"""
    if abs(value) > 1:
        value = max([min([value, 1]), -1])

    return math.asin(value)


def EulerFromMatrix(matrix, order):
    """Convert a rotation matrix into Euler angles based on the supplied axis order"""

    euler = [float('nan'), float('nan'), float('nan')]
    if any(map((lambda x: any(map(lambda y: math.isnan(y), x))), matrix)):
        return euler

    if order == 'xyz':
        euler[1] = _safe_arcsin(matrix[0][2])
        if abs(math.cos(euler[1])) > sys.float_info.epsilon * 100:
            euler[0] = math.atan2(-matrix[1][2], matrix[2][2])
            euler[2] = math.atan2(-matrix[0][1], matrix[0][0])
        else:
            if euler[1] > 0:
                euler[0] = math.atan2(matrix[1][0], matrix[1][1])
            else:
                euler[0] = -math.atan2(matrix[0][1], matrix[1][1])
            euler[2] = 0

    elif order == 'zyx':
        euler[1] = _safe_arcsin(-matrix[2][0])
        if abs(math.cos(euler[1])) > sys.float_info.epsilon * 100:
            euler[0] = math.atan2(matrix[2][1], matrix[2][2])
            euler[2] = math.atan2(matrix[1][0], matrix[0][0])
        else:
            if euler[1] > 0:
                euler[2] = math.atan2(-matrix[0][1], matrix[0][2])
            else:
                euler[2] = -math.atan2(-matrix[0][1], matrix[0][2])
            euler[0] = 0

    elif order == 'xzy':
        euler[2] = _safe_arcsin(-matrix[0][1])
        if abs(math.cos(euler[2])) > sys.float_info.epsilon * 100:
            euler[0] = math.atan2(matrix[2][1], matrix[1][1])
            euler[1] = math.atan2(matrix[0][2], matrix[0][0])
        else:
            if euler[2] > 0:
                euler[0] = math.atan2(-matrix[2][0], matrix[2][2])
            else:
                euler[0] = -math.atan2(-matrix[2][0], matrix[2][2])
            euler[1] = 0

    elif order == 'yzx':
        euler[2] = _safe_arcsin(matrix[1][0])
        if abs(math.cos(euler[2])) > sys.float_info.epsilon * 100:
            euler[0] = math.atan2(-matrix[1][2], matrix[1][1])
            euler[1] = math.atan2(-matrix[2][0], matrix[0][0])
        else:
            if euler[2] > 0:
                euler[1] = math.atan2(matrix[2][1], matrix[2][2])
            else:
                euler[1] = -math.atan2(matrix[2][1], matrix[2][2])
            euler[0] = 0

    elif order == 'yxz':
        euler[0] = _safe_arcsin(-matrix[1][2])
        if abs(math.cos(euler[0])) > sys.float_info.epsilon * 100:
            euler[1] = math.atan2(matrix[0][2], matrix[2][2])
            euler[2] = math.atan2(matrix[1][0], matrix[1][1])
        else:
            if euler[0] > 0:
                euler[1] = math.atan2(-matrix[0][1], matrix[0][0])
            else:
                euler[1] = -math.atan2(-matrix[0][1], matrix[0][0])
            euler[2] = 0

    elif order == 'zxy':
        euler[0] = _safe_arcsin(matrix[2][1])
        if abs(math.cos(euler[0])) > sys.float_info.epsilon * 100:
            euler[1] = math.atan2(-matrix[2][0], matrix[2][2])
            euler[2] = math.atan2(-matrix[0][1], matrix[1][1])
        else:
            if euler[0] > 0:
                euler[2] = math.atan2(matrix[0][2], matrix[0][0])
            else:
                euler[2] = -math.atan2(matrix[0][2], matrix[0][0])
            euler[1] = 0

    else:
        raise ValueError('Unsupported Euler angle combination')

    return euler


def MatrixFromEuler(angles, order):
    Matx = [[1, 0, 0],
            [0, math.cos(angles[0]), math.sin(angles[0])],
            [0, -math.sin(angles[0]), math.cos(angles[0])]]

    Maty = [[math.cos(angles[1]), 0, -math.sin(angles[1])],
            [0, 1, 0, ],
            [math.sin(angles[1]), 0, math.cos(angles[1])]]

    Matz = [[math.cos(angles[2]), math.sin(angles[2]), 0],
            [-math.sin(angles[2]), math.cos(angles[2]), 0],
            [0, 0, 1]]

    Mat1 = eval('Mat{}'.format(order[0]))
    Mat2 = eval('Mat{}'.format(order[1]))
    Mat3 = eval('Mat{}'.format(order[2]))

    def matmult(a, b):
        zip_b = zip(*b)
        if sys.version_info >= (3, 0):
            zip_b = list(zip_b)
        return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)) for col_b in zip_b] for row_a in a]

    RotMat = matmult(Mat3, matmult(Mat2, Mat1))
    return list(map(list, zip(*RotMat)))


def QuaternionFromMatrix(RotMat):
    """Calculates the quaternion representation of the rotation described by RotMat
    Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
    article "Quaternion Calculus and Fast Animation"."""

    Quaternion = [0.0] * 4
    Trace = float(sum([row[x] for x, row in enumerate(RotMat)]))

    if math.isnan(Trace):
        return [float('nan')] * 4

    if Trace > 0:
        Root = math.sqrt(Trace + 1)
        Quaternion[3] = 0.5 * Root
        Root = 0.5 / Root
        Quaternion[0] = (RotMat[2][1] - RotMat[1][2]) * Root
        Quaternion[1] = (RotMat[0][2] - RotMat[2][0]) * Root
        Quaternion[2] = (RotMat[1][0] - RotMat[0][1]) * Root
    else:
        Next = [1, 2, 0]
        i = 0
        if RotMat[1][1] > RotMat[0][0]:
            i = 1

        if RotMat[2][2] > RotMat[i][i]:
            i = 2

        j = Next[i]
        k = Next[j]

        Root = math.sqrt(RotMat[i][i] - RotMat[j][j] - RotMat[k][k] + 1)
        Quaternion[i] = 0.5 * Root
        Root = 0.5 / Root
        Quaternion[3] = (RotMat[k][j] - RotMat[j][k]) * Root
        Quaternion[j] = (RotMat[j][i] + RotMat[i][j]) * Root
        Quaternion[k] = (RotMat[k][i] + RotMat[i][k]) * Root

    norm = math.sqrt(sum(map(lambda n: n ** 2, Quaternion)))
    Quaternion = list(map(lambda n: n / norm, Quaternion))
    return Quaternion


def AngleAxisFromQuaternion(Quaternion):
    """Calculates the AngleAxis representation of the rotation described by a
    quaternion (x,y,z,w)"""
    if any(map(lambda x: math.isnan(x), Quaternion)):
        return [float('nan')] * 3

    imag = Quaternion[0:3]
    real = Quaternion[3]

    length = math.sqrt(sum(map(lambda x: x ** 2, imag)))
    if length < sys.float_info.epsilon * 100:
        AngleAxis = imag
    else:
        angle = 2 * math.atan2(length, real)
        AngleAxis = list(map(lambda x: angle / length * x, imag))

    return AngleAxis


def AngleAxisFromMatrix(RotMat):
    """Calculate the AngleAxis representation of the rotation described by a rotation matrix"""

    Quaternion = QuaternionFromMatrix(RotMat)
    AngleAxis = AngleAxisFromQuaternion(Quaternion)
    return AngleAxis


def RotationMatrixFromAngleAxis(k):
    """Convert angleaxis k to a 3x3 rotation matrix where k is a 3-vector
    defining the axis of rotation and norm(k) = angle of rotation about
    this axis"""

    R = [[0] * 3 for x in range(3)]
    fi = math.sqrt(sum(map(lambda n: n ** 2, k)))
    if fi < sys.float_info.epsilon * 100:
        for i in range(3):
            R[i][i] = 1
        return R

    x = k[0] / fi
    y = k[1] / fi
    z = k[2] / fi

    R[0][0] = math.cos(fi) + x ** 2 * (1 - math.cos(fi))
    R[0][1] = x * y * (1 - math.cos(fi)) - z * math.sin(fi)
    R[0][2] = x * z * (1 - math.cos(fi)) + y * math.sin(fi)

    R[1][0] = y * x * (1 - math.cos(fi)) + z * math.sin(fi)
    R[1][1] = math.cos(fi) + y ** 2 * (1 - math.cos(fi))
    R[1][2] = y * z * (1 - math.cos(fi)) - x * math.sin(fi)

    R[2][0] = z * x * (1 - math.cos(fi)) - y * math.sin(fi)
    R[2][1] = z * y * (1 - math.cos(fi)) + x * math.sin(fi)
    R[2][2] = math.cos(fi) + z ** 2 * (1 - math.cos(fi))

    return R
