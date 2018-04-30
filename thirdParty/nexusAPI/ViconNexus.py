import Client
import sys
import math
from NexusForcePlate import *
from NexusEyeTracker import *

class ViconNexus:
  # ViconNexus Creates a connection to Vicon Nexus for offline
  # data access
  #
  # Documentation and usage examples provided assume that an object named
  # vicon has been created to access the class methods
  #
  #  vicon = ViconNexus()

  def __init__( self ):
    # class constructor
      
    self.GenerateErrors = True
      
    # create our client object
    self.Client = Client.ViconNexusClient()
    self.Client.Connect('localhost')
   
  def Connect( self ):
    # Re-connect to the host application if we are not currently connected
    if ( not self.Client.IsConnected() ):
      self.Client.Connect('localhost')
  
  def DisplayCommandList( self ):
    # DisplayCommandList displays a list of commands available in the underlying Vicon SDK for the connected host application.      
    netCommands = self.Client.GetCommandList()
    for command in netCommands.Commands:
      print command

    if ( len(netCommands.Commands) == 0 ):
      if ( self.Client.IsConnected() ):
        print 'No commands found'
      else:
        print 'Host Application is not connected, unable to retrieve command list.'
  
  def DisplayCommandHelp( self, commandname ):
    # DisplayCommandHelp displays more detailed information about the specified Vicon SDK command 
    # This information is retrieved from the connected application
    commandInfo = self.Client.GetCommandInfo( commandname )
    print commandInfo.DefaultOutputString()
  
  def GetFrameCount( self ):
    # GetFrameCount retrieve the number of frames in the loaded trial      
    return self.Client.GetFrameCount()
  
  def GetFrameRate( self ):
    # GetFrameRate retrieves the base frame rate for the loaded trial      
    return self.Client.GetFrameRate()
  
  def GetTrialRange( self ):
    # GetTrialRange retrieves the starting and ending frame numbers of the
    # updateable range for the loaded trial      
    info = self.Client.GetTrialRange()
    return info.StartFrame, info.EndFrame

  def GetTrialRegionOfInterest( self ):
    # GetTrialRegionOfInterest retrieves the starting and ending frame numbers of the
    # user set region of interest for the loaded trial      
    info = self.Client.GetTrialRegionOfInterest()
    return info.StartFrame, info.EndFrame
      
  def SetTrialRegionOfInterest( self, startFrame, EndFrame ):
    # SetTrialRegionOfInterest sets the starting and ending frame numbers of the
    # user set region of interest for the loaded trial      
    result = self.Client.SetTrialRegionOfInterest( startFrame, EndFrame )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
      
  def GetTrialName( self ):
    # GetTrialName retrieves the name and path for the loaded trial      
    #
    # Returns
    #  path = string, path to the trial on disk
    #  name = string, name of the trial
    #
    # Usage Example: create a filename to be used for a user
    # generated output file
    #
    #  path, name = vicon.GetTrialName()
    #  MyFilename = path % name %'.MyFile'
    info = self.Client.GetTrialName()
    return unicode( info.Path.Value(), 'utf-8' ), unicode( info.Name.Value(), 'utf-8' )
  
  def GetSubjectNames( self ):
    # GetSubjectNames retrieve the names of the currently loaded subjects
    #
    # Returns
    #  names = list of strings, one for each subject
    #
    # Usage Example: list the names of the currently loaded subjects
    #
    #  subjects = vicon.GetSubjectNames()
    #  for subject in subjects:
    #    print subject 
    result = self.Client.GetSubjectNames()
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
      
    return map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Names )
  
  def GetMarkerNames( self, subject ):
    # GetMarkerNames retrieve the names of the markers associated with the specified subject
    #
    # Input
    #  subject = string, name of an existing subject
    # Returns
    #  names  = list of strings, one for each marker  
    #
    # Usage Example: Display the name of the first marker
    #
    #  markers = vicon.GetMarkerNames( 'Colin' )
    #  firstmarker = markers[0] # ???
    #  print firstmarker
    result = self.Client.GetMarkerNames( subject.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Names )
  
  def GetSegmentNames( self, subject ):
    # GetSegmentNames retrieve the names of the segments associated with the specified subject
    #
    # Input
    #  subject = string, name of an existing subject
    # Returns
    #  names  = list of strings, one for each segment  
    #
    # Usage Example: Display the name of the first segment
    #
    #  segments = vicon.GetSegmentNames( 'Colin' )
    #  firstsegment = segments[0]
    #  print firstsegment
    result = self.Client.GetSegmentNames( subject.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()

    return map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Names )
  
  def GetRootSegment( self, subject ):
    # GetRootSegment retrieve the name of the root segment associated with the specified subject
    #
    # Input
    #  subject = string, name of an existing subject
    # Returns
    #  name  = string, name of the root segment  
    #
    # Usage Example: Display the children of the root segment
    #
    #  root = vicon.GetRootSegment( 'Colin' )
    #  children = vicon.GetSegmentDetails( 'Colin', root )[1]
    #  for child in children:
    #    print child
    result = self.Client.GetRootSegment( subject.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
      
    return unicode( result.SegmentName.Value(), 'utf-8' )
  
  def GetSegmentDetails( self, subject, segment ):
    # GetSegmentDetails retrieves detailed information about a segment
    #
    # Input
    #  subject = string, name of an existing subject
    #  segment = string, name of an existing segment for the subject
    # Returns
    #  parent  = string, name of the parent segment
    #        *** the root segment will have a parent named 'World'
    #  children = list of strings, names of the child segments  
    #  markers = list of strings, names of the markers associated with the segment  
    #
    # Usage Example: Display the children of the root segment
    #
    #  root = vicon.GetRootSegment( 'Colin' )
    #  children = vicon.GetSegmentDetails( 'Colin', root )[1]
    #  for child in children:
    #    print child
    result = self.Client.GetSegmentDetails( subject.encode( 'utf-8' ), segment.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
      
    parent = unicode( result.Parent.Value(), 'utf-8' )
    children = map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Children )
    markers = map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Markers )

    return parent, children, markers
  
  def GetJointNames( self, subject ):
    # GetJointNames retrieve the names of the joints associated with the specified subject
    #
    # Input
    #  subject = string, name of an existing subject
    # Returns
    #  names  = list of strings, one for each joint  
    #
    # Usage Example: Display the name of the first joint
    #
    #  joints = vicon.GetJointNames( 'Colin' )
    #  firstjoint = joints[0]
    #  print firstjoint
    result = self.Client.GetJointNames( subject.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()

    return map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Names )
  
  def GetJointDetails( self, subject, joint ):
    # GetJointDetails retrieves detailed information about a joint
    #
    # Input
    #  subject = string, name of an existing subject
    #  joint  = string, name of an existing joint for the subject
    # Returns
    #  parent  = string, name of the parent segment
    #  child  = string, name of the child segment  
    #
    # Usage Example: Display information about the first joint
    #
    #  joints = vicon.GetJointNames( 'Colin' )
    #  firstjoint = joints[0]
    #  parent, child = vicon.GetJointDetails( 'Colin', firstjoint )
    #  JointDisplay = 'Joint: ' % firstjoint % ' ( ' % parent % ' - ' % child, ' )'
    #  print JointDisplay
    result = self.Client.GetJointDetails( subject.encode( 'utf-8' ), joint.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()

    parent = unicode( result.Parent.Value(), 'utf-8' )
    child  = unicode( result.Child.Value(), 'utf-8' )
    return parent, child
  
  def GetModelOutputNames( self, subject ):
    # GetModelOutputNames retrieve the names of the model outputs associated with the specified subject
    #
    # Input
    #  subject = string, name of an existing subject
    # Returns
    #  names  = list of strings, one for each model output  
    #
    # Usage Example: Display the name of the first model output
    #
    #  modeloutputs = vicon.GetModelOutputNames( 'Colin' )
    #  firstmodeloutput = modeloutputs[0]
    #  print firstmodeloutput
    result = self.Client.GetModelOutputNames( subject.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Names )
  
  def GetModelOutputDetails( self, subject, modelOutputName ):
    # GetModelOutputDetails retrieve detailed information about a model output
    #
    # Input
    #  subject     = string, name of an existing subject
    #  modelOutputName = string, name of an existing model output associated with the subject
    # Returns
    #  group   = string, name of the group the model output belongs to
    #  components = list of strings, list of component names for the model output
    #  types   = list of strings, list of the quantity types for each component
    #         *** use the DisplayCommandHelp method to get more information on
    #         quantity types that are valid for the connected application
    #
    # Usage Example: Create a new model output with the same properties as an existing model output
    #
    #  group, components, types = vicon.GetModelOutputDetails( 'Colin', 'LeftHipAngle' )
    #  vicon.CreateModelOutput( 'Colin', 'NewModelOutput', group, components, types )
    info = self.Client.GetModelOutputDetails( subject.encode( 'utf-8' ), modelOutputName.encode( 'utf-8' ) )
    if( info.Error() and self.GenerateErrors ):
      print >> sys.stderr, info.ResultString.Value()
       
    group = unicode( info.GroupName.Value(), 'utf-8' )
    components = map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  info.ComponentNames )
    types = map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  info.QuantityTypes )
    
    return group, components, types
  
  
  def GetSubjectParamNames( self, subject ):
    # GetSubjectParamNames retrieve the names of the static subject parameters associated with the specified subject
    #
    # Input
    #  subject = string, name of an existing subject 
    # Returns
    #  names  = list of strings, one for each subject parameter 
    #
    # Usage Example: Display all of the subject parameters
    #
    #  subjectparams = vicon.GetSubjectParamNames( 'Colin' )
    #  print subjectparams
    result = self.Client.GetSubjectParamNames( subject.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Names )
  
  
  def GetSubjectParamDetails( self, subject, param ):
    # GetSubjectParamDetails retrieve detailed information about a subject parameter
    #
    # Input
    #  subject = string, name of an existing subject  
    #  param  = string, name an existing subject parameter 
    # Returns
    #  value  = floating point number, current value of the subject parameter
    #  unit   = string, unit associated with the value
    #        *** use the DisplayCommandHelp method to get more information on
    #          units that are valid for the connected application
    #  default = floating point number, PRIOR value of the subject parameter
    #  required = logical, indication as to whether the subject parameter is a required parameter
    #  hasvalue = logical, indication as to whether the subject parameter has a value
    #
    # Usage Example: Display subject parameter details
    #
    #  value, unit, default, required = vicon.GetSubjectParamDetails( 'Colin', 'MyParam' )
    #  isRequired = ' Not Required'
    #  if( required ):
    #    isRequired = ' Required'
    #  
    #  SubjectParamInfo = 'MyParam = {0} [{1}] Default={2}, {3}'.format( value, unit, default, isRequired )
    #  print SubjectParamInfo 
    info = self.Client.GetSubjectParamDetails( subject.encode( 'utf-8' ), param.encode( 'utf-8' ) )
    if( info.Error() and self.GenerateErrors ):
      print >> sys.stderr, info.ResultString.Value()
       
    value = info.Value
    unit = unicode( info.Unit.Value(), 'utf-8' )
    default = info.DefaultValue
    required = info.Required
    hasvalue = info.HasValue
    
    return value, unit, default, required, hasvalue
  

  def GetSubjectParam( self, subject, param ):
    # GetSubjectParam retrieve the current value of a static subject parameter
    #
    # Input
    #  subject = string, name of an existing subject  
    #  param  = string, name of an existing subject parameter 
    # Returns
    #  value  = floating point number, current value of the subject parameter
    #  hasvalue = logical, indication as to whether the subject parameter has a value
    #
    # Usage Example: negate the value of a subject parameter
    #
    #  value = vicon.GetSubjectParam( 'Colin', 'MyParam' )
    #  value = value * (-1)
    #  vicon.SetSubjectParam( 'Colin', 'MyParam', value ) 
    data = self.Client.GetSubjectParam( subject.encode( 'utf-8' ), param.encode( 'utf-8' ) )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
       
    return data.Value, data.HasValue
  
  
  def SetSubjectParam( self, subject, param, value, bForce = False ):
    # SetSubjectParam set the current value of a static subject parameter
    #
    # Input
    #  subject = string, name of an existing subject  
    #  param  = string, name of an existing subject parameter 
    #  value  = floating point number, desired value of the subject parameter  
    #
    # Usage Example: negate the value of a subject parameter
    #
    #  value = vicon.GetSubjectParam( 'Colin', 'MyParam' )
    #  value = value * (-1)
    #  vicon.SetSubjectParam( 'Colin', 'MyParam', value )       
    result = self.Client.SetSubjectParam( subject.encode( 'utf-8' ), param.encode( 'utf-8' ), value, bForce )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
       
  
  
  def CreateSubjectParam( self, subject, param, value, unit, default, required ):
    # CreateSubjectParam create a new subject parameter for the specified subject. 
    # Subject parameter names must be unique within the subject.
    #
    # Input
    #  subject = string, name of an existing subject  
    #  param  = string, name of the new subject parameter 
    #  value  = floating point number, desired value of the subject parameter
    #  unit   = string, unit associated with the value
    #        *** use the DisplayCommandHelp method to get more information on
    #          units that are valid for the connected application
    #  default = floating point number, PRIOR value of the subject parameter
    #  required = logical, indication as to whether the subject parameter is a required parameter 
    #
    # Usage Example: create a new subject parameter
    #
    #  vicon.CreateSubjectParam( 'Colin', 'MyParam', 1.23, 'mm', 0, False )
    result = self.Client.CreateSubjectParam( subject.encode( 'utf-8' ), param.encode( 'utf-8' ), required, unit.encode( 'utf-8' ), value, default )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
       
  def GetUnlabeledCount( self ):
    # GetUnlabeledCount returns the number of unlabeled trajectories in the loaded trial
    return self.Client.GetUnlabeledCount()

  def GetUnlabeled( self, index ):
    # GetUnlabeled get all frames of data for the trial for the specified unlabeled trajectory.
    #
    # Input
    #  index  = index of the unlabeled trajectory to return
    # Returns
    #  x    = numerical(double) list, x-coordinates of the trajectory
    #  y    = numerical(double) list, y-coordinates of the trajectory
    #  z    = numerical(double) list, z-coordinates of the trajectory
    #  e    = logical list, T/F indication as to whether the data exists for each frame
    #
    # Usage Example: Display trajectory coordinate at frame 50 of unlabeled trajectory 25
    #
    #  trajX, trajY, trajZ, trajExists = vicon.GetUnlabeled( 25 )
    #  doesexist = ' - Missing Data'
    #  if( trajExists[49] ):
    #    doesexist = ' - exists'
    #  
    #  framedata = 'frame 50 = ({0}, {1}, {2}){3}'.format( trajX[49], trajY[49], trajZ[49], doesexist )
    #  print framedata       
    data = self.Client.GetUnlabeled( index )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    x = list(data.X)
    y = list(data.Y)
    z = list(data.Z)
    e = list(data.E)
    
    return x, y, z, e
    
  def HasTrajectory( self, subject, marker ):
    # HasTrajectory returns true if the specified marker is associated with a trajectory
    #
    # Input
    #    subject  = string, name of an existing subject   
    #    marker   = string, name of an existing marker
    # Returns
    #    exists   = T/F indication as to whether the specified marker is associated with a trajectory
    data = self.Client.HasTrajectory( subject.encode( 'utf-8' ), marker.encode( 'utf-8' ) );
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    exists = data.Exists
    return exists
    
  def GetTrajectory( self, subject, marker ):
    # GetTrajectory get all frames of data for the trial for the specified marker. 
    #
    # Input
    #  subject = string, name of an existing subject  
    #  marker  = string, name of an existing marker
    # Returns
    #  x    = numerical(double) list, x-coordinates of the trajectory
    #  y    = numerical(double) list, y-coordinates of the trajectory
    #  z    = numerical(double) list, z-coordinates of the trajectory
    #  e    = logical list, T/F indication as to whether the data exists for each frame
    #
    # Usage Example: Display trajectory coordinate at frame 50
    #
    #  trajX, trajY, trajZ, trajExists = vicon.GetTrajectory( 'Colin', 'C7' )
    #  doesexist = ' - Missing Data'
    #  if( trajExists[49] ):
    #    doesexist = ' - exists'
    #  
    #  framedata = 'frame 50 = ({0}, {1}, {2}){3}'.format( trajX[49], trajY[49], trajZ[49], doesexist )
    #  print framedata       
    data = self.Client.GetTrajectory( subject.encode( 'utf-8' ), marker.encode( 'utf-8' ) )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    x = list(data.X)
    y = list(data.Y)
    z = list(data.Z)
    e = list(data.E)
    
    return x, y, z, e
  
  
  def SetTrajectory( self, subject, marker, x, y, z, e ):
    # SetTrajectory update all of the data values for all frames in the trial for the specified marker.
    #
    # Input
    #  subject = string, name of an existing subject  
    #  marker  = string, name of an existing marker
    #  x    = numerical(double) list, x-coordinate of the trajectory
    #  y    = numerical(double) list, y-coordinate of the trajectory
    #  z    = numerical(double) list, z-coordinate of the trajectory
    #  e    = logical list, T/F indication as to whether the data exists for each frame
    #      
    # Usage Example: Put the trajectory at 0,0,0 for all frames
    #
    #  frameCount = vicon.GetFrameCount()
    #  trajX  = [0]*frameCount
    #  trajY  = [0]*frameCount
    #  trajZ  = [0]*frameCount
    #  exists = [True]*frameCount
    #  
    #  vicon.SetTrajectory('Colin','C7',trajX,trajY,trajZ,exists)
    
    for i in range( len( x ) ):
      if( any( map( math.isnan, [ x[i], y[i], z[i] ] ) ) ):
        e[i] = False
        x[i] = 0
        y[i] = 0
        z[i] = 0   
      
    result = self.Client.SetTrajectory( subject.encode( 'utf-8' ), marker.encode( 'utf-8' ), x, y, z, e )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
  
  
  def GetTrajectoryAtFrame( self, subject, marker, frame ):
    # GetTrajectoryAtFrame get trajectory data at a specific frame for the specified marker
    #
    # Input
    #  subject = string, name of an existing subject  
    #  marker  = string, name of an existing marker
    #  frame  = integer value, trial frame number as displayed in the application time bar
    # Returns
    #  x    = double value, x-coordinate of the trajectory
    #  y    = double value, y-coordinate of the trajectory
    #  z    = double value, z-coordinate of the trajectory
    #  e    = logical value, T/F indication as to whether the data exists for the frame
    #
    # Usage Example: Display trajectory coordinate at frame 50
    #
    #  [markerX, markerY, markerZ, markerExists] = vicon.GetTrajectoryAtFrame( 'Colin', 'C7', 50 )
    #  doesexist = ' - Missing Data'
    #  if( markerExists ):
    #    doesexist = ' - exists'
    #  
    #  framedata = ['frame 50 = ', num2str(markerX), ', ', num2str(markerY), ', ', num2str(markerZ), doesexist ]
    #  print framedata 
    data = self.Client.GetTrajectoryAtFrame( subject.encode( 'utf-8' ), marker.encode( 'utf-8' ), frame )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    x = data.X
    y = data.Y
    z = data.Z
    e = data.E
    
    return x, y, z, e
  
  
  def SetTrajectoryAtFrame( self, subject, marker, frame, x, y, z, e ):
    # SetTrajectoryAtFrame update the trajectory data values at a specific frame for the specified marker
    #
    # Input
    #  subject = string, name of an existing subject  
    #  marker  = string, name of an existing marker
    #  frame  = integer value, trial frame number as displayed in the application time bar
    #  x    = double value, x-coordinate of the trajectory
    #  y    = double value, y-coordinate of the trajectory
    #  z    = double value, z-coordinate of the trajectory
    #  e    = logical value, T/F indication as to whether the data exists for the frame
    #
    # Usage Example: Create a gap at frame 50
    #
    #  vicon.SetTrajectoryAtFrame( 'Colin', 'C7', 50, 0.0, 0.0, 0.0, False )
    
    if( any( map( math.isnan, [ x, y, z ] ) ) ):
      e = False
      x = 0
      y = 0
      z = 0
      
    result = self.Client.SetTrajectoryAtFrame( subject.encode( 'utf-8' ), marker.encode( 'utf-8' ), frame, x, y, z, e )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    

  def GetModelOutput( self, subject, modelOutputName ):
    # GetModelOutput get the data values for all components of a model output for all frames in the trial 
    #
    # Input
    #  subject     = string, name of an existing subject
    #  modelOutputName = string, name of an existing model output associated with the subject
    # Returns
    #  components   = numerical(double) NxM matrix where N is the number of components, M is the number of frames
    #  e        = logical list, T/F indication as to whether the data exists for each frame
    #
    # Usage Example: Copy the data from one model output to another
    #
    #  [data, exists] = vicon.GetModelOutput( 'Colin', 'LeftHipAngle' )
    #  vicon.SetModelOutput( 'Colin', 'NewAngle', data, exists )      
    data = self.Client.GetModelOutput( subject.encode( 'utf-8' ), modelOutputName.encode( 'utf-8' ) )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    componentCount = data.Data.size()
    components = [ list(data.Data[x]) for x in xrange(componentCount) ]
    e = list(data.E)
    
    return components, e
   
  
  def SetModelOutput( self, subject, modelOutputName, components, e ):
    # SetModelOutput set the data values for all components of a model output for all frames in the trial 
    #
    # Input
    #  subject     = string, name of the subject
    #  modelOutputName = string, name of an existing model output associated with the subject
    #  components   = numerical(double) NxM matrix where N is the number of components, M is the number of frames
    #  e        = logical list, T/F indication as to whether the data exists for each frame 
    #
    # Usage Example: Copy the data from one model output to another
    #
    #  [data, exists] = vicon.GetModelOutput( 'Colin', 'LeftHipAngle' )
    #  vicon.SetModelOutput( 'Colin', 'NewAngle', data, exists )
    
    componentDict = Client.map_ui_vd()
    i = 0
    for row in components:
      Values = Client.vectord()
      if( any( map( math.isnan, row ) ) ):
        e[i] = False
        row = [ 0 for x in row ]
          
      for value in row:
        Values.push_back( value )
      componentDict[i] = Values
      i += 1
      
    exists = Client.vectorb()
    for val in e:
      exists.push_back( val )
    
    result = self.Client.SetModelOutput( subject.encode( 'utf-8' ), modelOutputName.encode( 'utf-8' ), componentDict, exists )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
      
  
  def GetModelOutputAtFrame( self, subject, modelOutputName, frame ):
    # GetModelOutputAtFrame get the data values for all components of a model output at a specific frame      
    #
    # Input
    #  subject     = string, name of an existing subject
    #  modelOutputName = string, name of an existing model output associated with the subject
    #  frame      = integer value, trial frame number as displayed in the application time bar
    # Returns
    #  components   = numerical(double) list, one value for each component
    #  e        = logical value, T/F indication as to whether the data exists for the frame
    #
    # Usage Example: offset model output data by 100.0 at frame 50
    #
    #  data, exists = vicon.GetModelOutputAtFrame( 'Colin', 'NewAngle', 50 )
    #  for value in data:
    #    value = value + 100.0
    #  
    #  vicon.SetModelOutputAtFrame( 'Colin', 'NewAngle', 50, data, True )         
    data = self.Client.GetModelOutputAtFrame( subject.encode( 'utf-8' ), modelOutputName.encode( 'utf-8' ), frame )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    components = list(data.Data)
    e = data.E
  
    return components, e
  
  def SetModelOutputAtFrame( self, subject, modelOutputName, frame, components, e ):
    # SetModelOutputAtFrame set the data values for all components of a model output at a specific frame
    #
    # Input
    #  subject     = string, name of an existing subject
    #  modelOutputName = string, name of an existing model output associated with the subject
    #  frame      = integer value, trial frame number as displayed in the application time bar
    #  components   = numerical(double) list, one value for each component
    #  e        = logical value, T/F indication as to whether the data exists for the frame      
    #
    # Usage Example: offset model output data by 100.0 at frame 50
    #
    #  data, exists = vicon.GetModelOutputAtFrame( 'Colin', 'NewAngle', 50 )
    #  for value in data:
    #    value = value + 100.0
    #  
    #  vicon.SetModelOutputAtFrame( 'Colin', 'NewAngle', 50, data, True )
    
    if( any( map( math.isnan, components ) ) ):
      e = False
      components = [ 0 for x in components ]
      
    result = self.Client.SetModelOutputAtFrame( subject.encode( 'utf-8' ), modelOutputName.encode( 'utf-8' ), frame, components, e )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
  
  
  def CreateModelOutput( self, subject, modelOutputName, groupName, componentNames, QuantityTypes ):
    # CreateModelOutput create a new model output
    # Add data to the newly created model output using SetModelOutput or SetModelOutputAtFrame
    #
    # Input
    #  subject     = string, name of an existing subject
    #  modelOutputName = string, name of new model output associated with the subject
    #  group   = string, name of the group the model output belongs to
    #  components = list of strings, list of component names for the model output
    #  types   = list of strings, list of the quantity types for each component
    #         *** use the DisplayCommandHelp method to get more information on
    #         quantity types that are valid for the connected application
    #
    # Usage Example: Create model output matching the Plug-in Gait Bone HED
    #
    #  BonesNames = ['RX','RY','RZ','TX','TY','TZ','SX','SY','SZ']
    #  BonesTypes = ['Angle','Angle','Angle','Length','Length','Length','Length','Length','Length']
    #  vicon.CreateModelOutput( 'Patricia', 'HED', 'Plug-in Gait Bones', BonesNames, BonesTypes )
    
    SafeComps = []
    for name in componentNames:
      SafeComps.append( self.Client.AllocString( name.encode( 'utf-8' ) ) )

    SafeQuantities = []
    for Type in QuantityTypes:
      SafeQuantities.append( self.Client.AllocString( Type.encode( 'utf-8' ) ) )

    result = self.Client.CreateModelOutput( subject.encode( 'utf-8' ), modelOutputName.encode( 'utf-8' ), groupName, SafeComps, SafeQuantities )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
  

  def CreateModeledMarker( self, subject, modelOutputName ):
    # CreateModeledMarker creates a new model output that can be displayed in the application workspace
    # Add data to the newly created model output using SetModelOutput or SetModelOutputAtFrame
    #
    # Input
    #  subject     = string, name of an existing subject
    #  modelOutputName = string, name of new model output associated with the subject
    #
    # Usage Example: Create a midpoint 
    #
    #  vicon.CreateModeledMarker( 'Colin', 'Midpoint' )
    #  kneeX, kneeY, kneeZ, kneeExists = vicon.GetTrajectory( 'Colin', 'RKNE' )
    #  ankX, ankY, ankZ, ankExists = vicon.GetTrajectory( 'Colin', 'RANK' )
    #  data, exists = vicon.GetModelOutput( 'Colin', 'Midpoint' )
    #  framecount = vicon.GetFrameCount()
    #  for i in xrange(framecount):
    #    if kneeExists[i] and ankExists[i]
    #      exists[i] = True
    #      data[0][i] = ( kneeX[i] + ankX[i] ) / 2
    #      data[1][i] = ( kneeY[i] + ankY[i] ) / 2
    #      data[2][i] = ( kneeZ[i] + ankZ[i] ) / 2
    #    else:
    #      exists[i] = False
    #    
    #  
    #  vicon.SetModelOutput( 'Colin', 'Midpoint', data, exists )      
    XYZNames = ['X','Y','Z']
    Names = []
    for name in XYZNames:
      Names.append( self.Client.AllocString( name ) )
    Types = ['Length','Length','Length']
    SafeTypes = []
    for Type in Types:
      SafeTypes.append( self.Client.AllocString( Type ) )
    result = self.Client.CreateModelOutput( subject.encode( 'utf-8' ), modelOutputName.encode( 'utf-8' ), 'Modeled Markers', Names, SafeTypes )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
      
  
  def ClearAllEvents( self ):
    # ClearAllEvents will delete all events currently defined in the loaded trial
    result = self.Client.ClearAllEvents()
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
          
  
  
  def GetEvents( self, subject, context, event ):
    # GetEvent will return all of the events that match the specified subject, context, and event type
    #
    # Input
    #  subject     = string, name of an existing subject
    #  context     = string, name of the context. 
    #           *** use the DisplayCommandHelp method to get more information on
    #           context names that are valid for the connected application
    #  event      = string, name of the event type. 
    #           *** use the DisplayCommandHelp method to get more information on
    #           event type names that are valid for the connected application
    # Returns
    #  frames     = integer list, list of the event frame numbers
    #  offsets     = double list, offset (in seconds) from the beginning of the frame 
    #           to the event occurrence for each event
    #           The value should be in the range of 0.00 to 1/FrameRate
    #
    # Usage Example: 
    #
    #  vicon.CreateAnEvent( 'Patricia', 'Foot Strike', 137, 0.0 )
    data = self.Client.GetEvents( subject.encode( 'utf-8' ), context.encode( 'utf-8' ), event.encode( 'utf-8' ) )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
    
    frames = list(data.FrameNumbers)
    offsets = list(data.FrameOffsets)
    
    return frames, offsets

   
  
  def CreateAnEvent( self, subject, context, event, frame, offset ):
    # CreateAnEventt create a new event at the specifed ( frame + offset )
    #
    # Input
    #  subject     = string, name of an existing subject
    #  context     = string, name of the context. 
    #           *** use the DisplayCommandHelp method to get more information on
    #           context names that are valid for the connected application
    #  event      = string, name of the event type. 
    #           *** use the DisplayCommandHelp method to get more information on
    #           event type names that are valid for the connected application
    #  frame      = integer value, trial frame number as displayed in the application time bar
    #  offset     = double value, offset (in seconds) from the beginning of the frame to the event occurrence
    #           The value should be in the range of 0.00 to 1/FrameRate
    #
    # Usage Example: 
    #
    #  vicon.CreateAnEvent( 'Patricia', 'Foot Strike', 137, 0.0 )
    result = self.Client.CreateAnEvent( subject.encode( 'utf-8' ), context.encode( 'utf-8' ), event.encode( 'utf-8' ), frame, offset )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
      
  
  def RunPipeline( self, pipeline, location, timeout ):
    # RunPipeline will run a pipeline in the connected host application. This command may be used
    # from within Python but can not be used in a script that is executing from the application.
    # The pipeline will fail to run if another pipeline is already in progress.  
    #
    # Input
    #    pipeline        = string, name of an existing pipeline.
    #    location        = string, location of the pipeline. Pass a blank string to use the default
    #                      searching mechanism to locate the pipeline. Valid options when specifying the 
    #                      pipeline location are Private, Shared or System.
    #    timeout         = integer value, timeout value (in seconds) used to return control in the case that
    #                      the pipeline does not complete in a timely fashion. It is important to note that
    #                      the timeout value has no effect in the host application itself and the pipeline
    #                      may continue to run after the time allocated has expired.
    #
    # Usage Example: 
    #
    #  vicon.RunPipeline( 'Reconstruct', '', 45 )
    result = self.Client.RunPipeline( pipeline.encode( 'utf-8' ), location.encode( 'utf-8' ), timeout )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
   
  
  def OpenTrial( self, trial, timeout ):
    # OpenTrial will open a trial in the connected host application. This command may be used
    # from within Python but can not be used in a script that is executing from the application.
    # The trial will fail to load if a pipeline is running, the application is in live mode, the 
    # ENF file for the trial can not be located, or the application is in a state where trial 
    # opening is prohibited.
    # 
    # If unsaved data is currently loaded in the application, the application may prompt you to save the data.
    # To prevent the application prompt, you should save the current trial data prior to attempting to 
    # load another trial.
    #
    # Input
    #  trial      = string, name of an existing trial including its full path, excluding any file extensions.
    #  timeout     = integer value, timeout value (in seconds) used to return control in the case that
    #           the trial does not open in a timely fashion. It is important to note that
    #           the timeout value has no effect in the host application itself and the trial
    #           may continue to load after the time allocated has expired.
    #
    # Usage Example: 
    #
    #  vicon.OpenTrial( 'C:\Users\Public\Documents\Vicon\Nexus Sample Data\Colin\Walking Trials\Walking 2', 30 )
    result = self.Client.OpenTrial( trial.encode( 'utf-8' ), timeout )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
   
  
  def SaveTrial( self, timeout ):
    # SaveTrial will save the trial currently open in the connected host application to disk.
    # This command may be used from within Python but can not be used in a script that is 
    # executing from the application.
    # The trial will fail to save if a pipeline is running or the application is is live mode.
    #
    # Input
    #  timeout     = integer value, timeout value (in seconds) used to return control in the case that
    #           the trial does not save in a timely fashion. It is important to note that
    #           the timeout value has no effect in the host application itself and the trial
    #           may continue to save after the time allocated has expired.
    #
    # Usage Example: 
    #
    #  vicon.SaveTrial( 30 )
    result = self.Client.SaveTrial( timeout )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
     
   
  
  def GetDeviceIDs( self ):
    # GetDeviceIDs retrieve a list of the unique identifiers of the analog devices
    # The DeviceID is used for all device communication. A device may also be 'named'
    # although having a device name is not a requirement. 
    #
    # Returns
    #  deviceIDs = list of unsigned ints, one for each device
    #
    # Usage Example: retrieve the list of device identifiers
    #
    #  devices = vicon.GetDeviceIDs()
    #  for device in devices:
    #    print device
    result = self.Client.GetDeviceIDs()
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return list(result.IDs)
  
  
  def GetDeviceNames( self ):
    # GetDeviceNames retrieve a list device names. 
    # Device names are not required, access to device data is done using the DeviceID 
    #
    # Returns
    #  deviceNames = list of strings, one for each named device
    #
    # Usage Example: 

    result = self.Client.GetDeviceNames()
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  result.Names )
   
  
  def GetDeviceIDFromName( self, name ):
    # GetDeviceIDFromName will search for the specified device using the device 
    # name to make the match.
    # The DeviceID is used for all device communication.      
    #
    # Input
    #  name   = string, name of a known device, device names are case sensitive     
    # Returns
    #  deviceID = string, DeviceID of the device with name 'name'
    #
    # Usage Example: retrieve the DeviceID for a named forceplate
    #
    #  DeviceID = vicon.GetDeviceIDFromName( 'MyForcePlate' )
    result = self.Client.GetDeviceIDFromName( name.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return result.DeviceID
  
  
  def GetDeviceDetails( self, deviceID ):
    # GetDeviceDetails will return detailed information about a specific device.
    # Every device will have associated device outputs. Communication with 
    # the device outputs is done by using a DeviceOutputID to identify the 
    # specific device output.
    #
    # Input
    #  deviceID    = string, DeviceID of the device you are interested in     
    # Returns
    #  name      = string, name of the device (may be blank)
    #  type      = string, device type. Valid options are
    #           'ForcePlate', 'EyeTracker', 'Other'
    #  rate      = double value, rate at which the device runs
    #  deviceOutputIDs = unsigned int, list of the DeviceOutputIDs
    #  forceplate   = NexusForcePlate, additional info if the device 
    #           is a force plate
    #  eyetracker   = NexusEyeTracker, additional info if the device
    #           is an eye tracker
    #
    # Usage Example: Display the name and type of a device
    #
    #  name, type, rate, deviceOutputIDs, forceplate, eyetracker = vicon.GetDeviceDetails( '1' )
    #  DeviceDisplay = 'DeviceID: {0} is named {1} and it is a {2} device' ].format( deviceID, name, type )
    #  print DeviceDisplay 
    info = self.Client.GetDeviceDetails( deviceID )
    if( info.Error() and self.GenerateErrors ):
      print >> sys.stderr, info.ResultString.Value() 
          
    name = unicode( info.Name.Value(), 'utf-8' )
    type = unicode( info.Type.Value(), 'utf-8' )
    rate = info.Rate
    deviceOutputIDs = list(info.DeviceOutputIDs)
    forceplate = NexusForcePlate()
    forceplate.LocalR = list(info.FP_LocalR())
    forceplate.LocalT = list(info.FP_LocalT())
    forceplate.WorldR = list(info.FP_WorldR())
    forceplate.WorldT = list(info.FP_WorldT())
    forceplate.LowerBounds = list(info.FP_LowerBounds())
    forceplate.UpperBounds = list(info.FP_UpperBounds())
    forceplate.Context = info.FP_Context()
    eyetracker = NexusEyeTracker()
    eyetracker.SubjectName = info.ET_SubjectName()
    eyetracker.SegmentName = info.ET_SegmentName()
    eyetracker.EyePoseT = list(info.ET_EyePoseT())
    eyetracker.EyePoseR = list(info.ET_EyePoseR())
    eyetracker.Offset = list(info.ET_Offset())
  
    return name, type, rate, deviceOutputIDs, forceplate, eyetracker
  
  def GetDeviceOutputIDFromName( self, deviceID, name ):
    # GetDeviceOutputIDFromName will search device 'deviceID' for the named device output 'name'
    # The DeviceOutputID is used for all device output communication.      
    #
    # Input
    #  deviceID    = unsigned int, DeviceID of and existing device
    #  name      = string, name of a known device output, names are case sensitive  
    # Returns
    #  deviceOutputID = unsigned int, DeviceOutputID of the device with name 'name'
    #
    # Usage Example: retrieve the DeviceOutputID for a named named output
    #
    #  DeviceOutputID = vicon.GetDeviceOutputIDFromName( 1, 'Force' )
    result = self.Client.GetDeviceOutputIDFromName( deviceID, name.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return result.DeviceOutputID
   
  
  def GetDeviceChannelIDFromName( self, deviceID, deviceOutputID, name ):
    # GetDeviceChannelIDFromName will search the specified device
    # output and return the channel ID with the name 'name'.
    # The deviceChannelID is used for all device output channelcommunication.      
    #
    # Input
    #  deviceID    = unsigned int, DeviceID of and existing device
    #  deviceOutputID = unsigned int, DeviceOutputID of the device output you are interested in
    #  name      = string, name of a known channel, names are case sensitive  
    # Returns
    #  channelID    = unsigned int, ChannelID of the device with name 'name'
    #
    # Usage Example: 
    #
    result = self.Client.GetDeviceChannelIDFromName( deviceID, deviceOutputID, name.encode( 'utf-8' ) )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
    return result.ChannelID
        
  
  def GetDeviceOutputDetails( self, deviceID, deviceOutputID ):
    # GetDeviceOutputDetails will return detailed information about a specific device.
    # output. 
    #
    # Input
    #  deviceID    = unsigned int, DeviceID of and existing device
    #  deviceOutputID = unsigned int, DeviceOutputID of the device output you are interested in
    # Returns
    #  name      = string, name of the device (may be blank)
    #  type      = string, device output type. 
    #  unit      = string, unit name
    #  ready      = logical value, indication of whether or not the output is in the ready state
    #  channelNames  = string list, list of channel names associated with the output,
    #           channel names are not required, data access is acheived using the channelID
    #  channelIDs   = unsigned int list, list of channel IDs associated with the output
    #  
    # Usage Example: 
    #
    info = self.Client.GetDeviceOutputDetails( deviceID, deviceOutputID )
    if( info.Error() and self.GenerateErrors ):
      print >> sys.stderr, info.ResultString.Value() 
          
    name =  unicode( info.Name.Value(), 'utf-8' )
    type =  unicode( info.Type.Value(), 'utf-8' )
    unit =  unicode( info.UnitName.Value(), 'utf-8' )
    ready = info.Ready
    channelNames = map( lambda x: unicode( ViconNexus._GetSafeStringValue(x), 'utf-8'),  info.ChannelNames )    
    channelIDs = list(info.ChannelIDs)
      
    return name, type, unit, ready, channelNames, channelIDs
  
  def GetDeviceChannel( self, deviceID, deviceOutputID, channelID ):
    # GetDeviceChannel will return a single channel of device data 
    # from the channel identified by deviceID:deviceOutputID:channelID      
    # A device can have multiple device outputs and each device output
    # can have multiple channels associated with it
    #
    # Devices can run at different rates than the trial data.
    # Channel data could have multiple samples for each trial frame. All samples for a channel are output.
    # For data that has a higher sample rate than the trial frame rate, channel data can be interpreted as:
    #  frame1[sample1], frame1[sample2], ... frame1[sampleN], ... frameN[sample1], frameN[sample2], ... frameN[sampleN]
    #
    # Input
    #  deviceID     = unsigned int, DeviceID of and existing device
    #  deviceOutputID  = unsigned int, DeviceOutputID of the device output you are interested in
    #  channelID    = unsigned int, ID of the channel
    # Returns
    #  channelData   = numerical(double) list, component data list of size NumberOfFrames * samplesPerFrame
    #  ready      = logical, T/F indication as to whether or not the device output is in the ready state
    #           if the device output is not in the ready state, there will not be any valid data
    #           associated with this device output component
    #  rate      = double value, sample rate of the channel data 
    #
    # Usage Example: 
    #
   
    data = self.Client.GetDeviceChannel( deviceID, deviceOutputID, channelID )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
    
    channelData = list(data.Data)
    ready = data.Ready
    rate = data.Rate
    
    return channelData, ready, rate
  
  def GetDeviceChannelAtFrame( self, deviceID, deviceOutputID, channelID, frame ):
    # GetDeviceChannelAtFrame will retrieve a single frame of data 
    # from the channel identified by deviceID:deviceOutputID:channelID
    # A device can have multiple device outputs and each device output
    # can have multiple channels associated with it
    #
    # Devices can run at different rates than the trial data.
    # Channel data could have multiple samples for each trial frame. All samples for a channel are output.
    # Channel data list will be in the format:
    #  [sample1], [sample2], ... [sampleN]
    #
    # Input
    #  deviceID     = unsigned int, DeviceID of and existing device
    #  deviceOutputID  = unsigned int, DeviceOutputID of the device output you are interested in
    #  channelID    = unsigned int, ID of the channel
    #  frame      = integer value, trial frame number as displayed in the application time bar
    # Returns
    #  channelData   = numerical(double) list, component data list for the frame of size samplesPerFrame
    #  ready      = logical, T/F indication as to whether or not the device output is in the ready state
    #           if the device output is not in the ready state, there will not be any valid data
    #           associated with this device output component
    #  rate      = double value, sample rate of the channel data 
    #
    # Usage Example: 
    #
   
    data = self.Client.GetDeviceChannelAtFrame( deviceID, deviceOutputID, channelID, frame )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    channelData = list(data.Data)
    ready = data.Ready
    rate = data.Rate
    
    return channelData, ready, rate
  
  def GetDeviceChannelGlobal( self, deviceID, deviceOutputID, channelID ):
    # GetDeviceChannel will return a single channel of global device data 
    # from the channel identified by deviceID:deviceOutputID:channelID  
    # A device can have multiple device outputs and each device output
    # can have multiple channels associated with it
    #
    # Devices can run at different rates than the trial data.
    # Channel data could have multiple samples for each trial frame. All samples for a channel are output.
    # For data that has a higher sample rate than the trial frame rate, channel data can be interpreted as:
    #  frame1[sample1], frame1[sample2], ... frame1[sampleN], ... frameN[sample1], frameN[sample2], ... frameN[sampleN]
    #
    # Input
    #  deviceID     = unsigned int, DeviceID of and existing device
    #  deviceOutputID  = unsigned int, DeviceOutputID of the device output you are interested in
    #  channelID    = unsigned int, ID of the channel
    # Returns
    #  channelData   = numerical(double) list, component data list of size NumberOfFrames * samplesPerFrame
    #  ready      = logical, T/F indication as to whether or not the device output is in the ready state
    #           if the device output is not in the ready state, there will not be any valid data
    #           associated with this device output component
    #  rate      = double value, sample rate of the channel data 
    #
    # Usage Example: 
    #
   
    data = self.Client.GetDeviceChannelGlobal( deviceID, deviceOutputID, channelID )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
    
    channelData = list(data.Data)
    ready = data.Ready
    rate = data.Rate
      
    return channelData, ready, rate
  
  def GetDeviceChannelGlobalAtFrame( self, deviceID, deviceOutputID, channelID, frame ):
    # GetDeviceChannelGlobalAtFrame will retrieve a single frame of global data 
    # from a channel identified by deviceID:deviceOutputID:channelID
    # A device can have multiple device outputs and each device output
    # can have multiple channels associated with it
    #
    # Devices can run at different rates than the trial data.
    # Channel data could have multiple samples for each trial frame. All samples for a channel are output.
    # Channel data list will be in the format:
    #  [sample1], [sample2], ... [sampleN]
    #
    # Input
    #  deviceID     = unsigned int, DeviceID of and existing device
    #  deviceOutputID  = unsigned int, DeviceOutputID of the device output you are interested in
    #  channelID    = unsigned int, ID of the channel
    #  frame      = integer value, trial frame number as displayed in the application time bar
    # Returns
    #  channelData   = numerical(double) list, component data list for the frame of size samplesPerFrame
    #  ready      = logical, T/F indication as to whether or not the device output is in the ready state
    #           if the device output is not in the ready state, there will not be any valid data
    #           associated with this device output component
    #  rate      = double value, sample rate of the channel data 
    #
    # Usage Example: 
    #
   
    data = self.Client.GetDeviceChannelGlobalAtFrame( deviceID, deviceOutputID, channelID, frame )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    channelData = list( data.Data )
    ready = data.Ready
    rate = data.Rate
      
    return channelData, ready, rate
  
  def SetDeviceChannel( self, deviceID, deviceOutputID, channelID, channelData ):
    # SetDeviceChannel will update a single channel of device data for the specified 
    # deviceID:deviceOutputID:channelID combination
    # A device can have multiple device outputs and each device output
    # can have multiple channels associated with it
    #
    # You may not update frames of data that have been marked as
    # missing in the originally captured data, data for missing
    # frames will be presented with zero values when retrieving
    # data and input values for those frames will be ignored when
    # updating the device data, although it must be supplied.
    #
    # Devices can run at different rates than the trial data.
    # Channel data could have multiple samples for each data frame. 
    # Channel data must be supplied for all of the samples.
    # In the case that the channel sample rate is higher than the
    # trial frame rate channel data is interpreted as:
    #  frame1[sample1], frame1[sample2], ... frame1[sampleN], ... 
    #  frameN[sample1], frameN[sample2], ... frameN[sampleN]
    #
    # Input
    #  deviceID     = unsigned int, DeviceID of an existing device
    #  deviceOutputID  = unsigned int, DeviceOutputID of the device output 
    #  channelID    = unsigned int, ID of the channel
    #  channelData   = numerical(double) list, channel data list of size NumberOfFrames * samplesPerFrame
    #
    # Usage Example: 
    #
   
    result = self.Client.SetDeviceChannel( deviceID, deviceOutputID, channelID, channelData )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
    
      
  
  def SetDeviceChannelAtFrame( self, deviceID, deviceOutputID, channelID, frame, channelData ):
    # SetDeviceChannelAtFrame will update a single frame for a single channel
    # of device data for the specified deviceID:deviceOutputID combination
    # A device can have multiple channels associated with it
    #
    # You may not update frames of data that have been marked as
    # missing in the originally captured data, data for missing
    # frames will be presented with zero values when retrieving
    # data and input values for those frames will be ignored when
    # updating the device data, although it must be supplied.
    #
    # CAUTION: updating multiple frames of device data using this
    # def may execute noticably slower than using a single
    # call to SetDeviceChannel supplying sample data for each trial frame
    # even if they have not changed depending on the number of samples you
    # are updating. 
    #
    # Devices can run at different rates than the trial data.
    # Channel data could have multiple samples for each trial frame. 
    # Channel data list must be in the format:
    #  [sample1], [sample2], ... [sampleN]
    #
    # Input
    #  deviceID     = unsigned int, DeviceID of and existing device
    #  deviceOutputID  = unsigned int, DeviceOutputID of the device output you are interested in
    #  channelID    = unsigned int, ID of the channel
    #  frame      = integer value, trial frame number as displayed in the application time bar
    #  channelData   = numerical(double) list, component data list for the frame of size samplesPerFrame
    #
    # Usage Example: 
    #
   
    result = self.Client.SetDeviceChannelAtFrame( deviceID, deviceOutputID, channelID, frame, channelData )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()
          
  def SubmitSplineTrajectory( self, frames, xData, yData, zData, frameRate ):
    # SubmitSplineTrajectory initialize a spline object
    #
    # Input
    #    frames    = numerical(int) list, frame numbers of the supplied data
    #    x         = numerical(double) list, x-coordinate of the trajectory
    #    y         = numerical(double) list, y-coordinate of the trajectory
    #    z         = numerical(double) list, z-coordinate of the trajectory
    #    frameRate = double, frame rate of the supplied data
    #            
    # Usage Example:
    #
    #    frameRate = vicon.GetFrameRate()
    #    startFrame, endFrame = vicon.GetTrialRange()
    #    frames = range( startFrame, endFrame + 1 )
    #    [trajX, trajY, trajZ, e] = vicon.GetTrajectory( 'Colin', 'LKNE' )
    #    # get valid indices; we only want to submit good values here!            
    #    validIndices = [ x for x, y in enumerate(e) if y ]
    #    validF = [ frames[ Index ] for Index in validIndices ]
    #    validX = [ trajX[ Index ] for Index in validIndices ]
    #    validY = [ trajY[ Index ] for Index in validIndices ]
    #    validZ = [ trajZ[ Index ] for Index in validIndices ]
    #    vicon.SubmitSplineTrajectory(validF, validX, validY, validZ, frameRate)  
    ClientFrames = Client.vectori()
    for frame in frames:
      ClientFrames.push_back( frame )
      
    ClientXData = Client.vectord()
    for x in xData:
      ClientXData.push_back( x )
      
    ClientYData = Client.vectord()
    for y in yData:
      ClientYData.push_back( y )
      
    ClientZData = Client.vectord()
    for z in zData:
      ClientZData.push_back( z )
      
    result = self.Client.SubmitSplineTrajectory( ClientFrames, ClientXData, ClientYData, ClientZData, frameRate )
    if( result.Error() and self.GenerateErrors ):
      print >> sys.stderr, result.ResultString.Value()  
      
  def GetSplineResults( self, derivative ):
    # GetSplineResults return derivative values from an initialized spline object. 
    #  
    # Input
    #    frames    = numerical(int) list, frame numbers of the supplied data
    #    x         = numerical(double) list, x-coordinate of the trajectory
    #    y         = numerical(double) list, y-coordinate of the trajectory
    #    z         = numerical(double) list, z-coordinate of the trajectory
    #    frameRate = double, frame rate of the supplied data
    #            
    # Usage Example:
    #
    #    frameRate = vicon.GetFrameRate()
    #    startFrame, endFrame = vicon.GetTrialRange()
    #    frames = range( startFrame, endFrame + 1 )
    #    [trajX, trajY, trajZ, e] = vicon.GetTrajectory( 'Colin', 'LKNE' )
    #    # get valid indices; we only want to submit good values here!            
    #    validIndices = [ x for x, y in enumerate(e) if y ]
    #    validF = [ frames[ Index ] for Index in validIndices ]
    #    validX = [ trajX[ Index ] for Index in validIndices ]
    #    validY = [ trajY[ Index ] for Index in validIndices ]
    #    validZ = [ trajZ[ Index ] for Index in validIndices ]
    #    vicon.SubmitSplineTrajectory(validF, validX, validY, validZ, frameRate)  
    #
    #    accX, accY, accZ = vicon.GetSplineResults(2);  
  
    data = self.Client.GetSplineResults( derivative )
    if( data.Error() and self.GenerateErrors ):
      print >> sys.stderr, data.ResultString.Value()
          
    x = list(data.X)
    y = list(data.Y)
    z = list(data.Z)
    
    return x, y, z
          
          
  @staticmethod
  def _GetSafeStringValue( sString ):
    return sString.Value()




