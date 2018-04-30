class ViconUtils:
  # a collection of utilities
  
  def __init__( self ):
      # class constructor
      pass
    
  def CreateViconGaitModelOutputs( self, ViconNexus ):
    # create the model outputs formatted to match Vicon Plug-In_Gait
    # model outputs will be created for each subject in the workspace

    # get a list of the currently loaded subjects
    subjects = ViconNexus.GetSubjectNames()

    for subjectname in subjects:
  
      XYZNames = ['X','Y','Z']
  
      # Angles
      AnglesTypes = ['Angle','Angle','Angle']
      ViconNexus.CreateModelOutput( subjectname, 'LAbsAnkleAngle',   'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LAnkleAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LElbowAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LFootProgressAngles', 'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LHeadAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LHipAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LKneeAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LNeckAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LPelvisAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LShoulderAngles',   'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LSpineAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LThoraxAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LWristAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RAbsAnkleAngle',   'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RAnkleAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RElbowAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RFootProgressAngles', 'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RHeadAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RHipAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RKneeAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RNeckAngles',     'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RPelvisAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RShoulderAngles',   'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RSpineAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RThoraxAngles',    'Angles', XYZNames, AnglesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RWristAngles',    'Angles', XYZNames, AnglesTypes )
  
      # Forces
      ForcesTypes = ['Force','Force','Force']
      ForcesNormalizedTypes = ['ForceNormalized','ForceNormalized','ForceNormalized']
      ViconNexus.CreateModelOutput( subjectname, 'LAnkleForce',     'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LElbowForce',     'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LGroundReactionForce', 'Forces', XYZNames, ForcesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LHipForce',      'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LKneeForce',      'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LNeckForce',      'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LShoulderForce',    'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LWaistForce',     'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LWristForce',     'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RAnkleForce',     'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RElbowForce',     'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RGroundReactionForce', 'Forces', XYZNames, ForcesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RHipForce',      'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RKneeForce',      'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RNeckForce',      'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RNormalisedGRF',    'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RShoulderForce',    'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RWaistForce',     'Forces', XYZNames, ForcesNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RWristForce',     'Forces', XYZNames, ForcesNormalizedTypes )
  
      # Moments
      MomentsTypes = ['Torque','Torque','Torque']
      MomentsNormalizedTypes = ['TorqueNormalized','TorqueNormalized','TorqueNormalized']
      ViconNexus.CreateModelOutput( subjectname, 'LAnkleMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LElbowMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LGroundReactionMoment', 'Moments', XYZNames, MomentsTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LHipMoment',      'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LKneeMoment',      'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LNeckMoment',      'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LShoulderMoment',    'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LWaistMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LWristMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RAnkleMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RElbowMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RGroundReactionMoment', 'Moments', XYZNames, MomentsTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RHipMoment',      'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RKneeMoment',      'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RNeckMoment',      'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RShoulderMoment',    'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RWaistMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RWristMoment',     'Moments', XYZNames, MomentsNormalizedTypes )
  
      # Plug-in Gait Bones
      BonesNames = ['RX','RY','RZ','TX','TY','TZ','SX','SY','SZ']
      BonesTypes = ['Angle','Angle','Angle','Length','Length','Length','Length','Length','Length']
      ViconNexus.CreateModelOutput( subjectname, 'HED', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LCL', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LFE', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LFO', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LHN', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LHU', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LRA', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LTI', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LTO', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'PEL', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RCL', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RFE', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RFO', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RHN', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RHU', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RRA', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RTI', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RTO', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      ViconNexus.CreateModelOutput( subjectname, 'TRX', 'Plug-in Gait Bones', BonesNames, BonesTypes )
      
      # Powers
      PowersNormalizedTypes = ['PowerNormalized','PowerNormalized','PowerNormalized']
      ViconNexus.CreateModelOutput( subjectname, 'LAnklePower',  'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LElbowPower',  'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LHipPower',   'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LKneePower',   'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LNeckPower',   'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LShoulderPower', 'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LWaistPower',  'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'LWristPower',  'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RAnklePower',  'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RElbowPower',  'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RHipPower',   'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RKneePower',   'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RNeckPower',   'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RShoulderPower', 'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RWaistPower',  'Powers', XYZNames, PowersNormalizedTypes )
      ViconNexus.CreateModelOutput( subjectname, 'RWristPower',  'Powers', XYZNames, PowersNormalizedTypes )

  def Globalise( self, point, WorldPoseR, WorldPoseT ):
    # given a world pose, globalize the point   
    # WorldPoseR - World rotation matrix (row major format, 9 elements)
    # WorldPoseT - World translation in mm (3 elements)
            
    RotMat = [[0]*3]*3
    idx = 0
    for i in range(3):
      for j in range(3):
        RotMat[i][j] = WorldPoseR[idx]
        idx = idx + 1

    inputData = point
    TVector = WorldPoseT
    globalisedPoint = [0]*3
    for i in range(3):
      for j in range(3):
        globalisedPoint[i] = globalisedPoint[i] + (RotMat[i][j] * inputData[j])
    for i in range(3):
      globalisedPoint[i] = globalisedPoint[i] + TVector[i]

    return globalisedPoint
    
        
  def Localise( self, point, WorldPoseR, WorldPoseT ):
    # given a world pose, localize the point   
    # WorldPoseR - World rotation matrix (row major format, 9 elements)
    # WorldPoseT - World translation in mm (3 elements)
            
    RotMat = [[0]*3]*3
    idx = 0
    for i in range(3):
      for j in range(3):
        RotMat[i][j] = WorldPoseR[idx]
        idx = idx + 1

    inputData = point
    TVector = WorldPoseT

    # negate matrix
    NegRotMat = [[0]*3]*3
    for i in range(3):
      for j in range(3):
        NegRotMat[i][j] = RotMat[i][j]*-1

    temp =  [0]*3    
    for i in range(3):
      for j in range(3):
        temp[i] = temp[i] + (NegRotMat[j][i] * TVector[j])
    TVector = temp

    # localise the point
    localisedPoint = [0]*3 
    for i in range(3):
      for j in range(3):
        localisedPoint[i] = localisedPoint[i] + (RotMat[j][i] * inputData[j])
    for i in range(3):
      localisedPoint[i] = localisedPoint[i] + TVector[i] 

    return localisedPoint  
      
