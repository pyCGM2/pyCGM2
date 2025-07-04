<?xml version="1.1" encoding="UTF-8" standalone="no" ?>
<Pipeline>

  <Entry DisplayName="Combined Processing" Enabled="0" OperationId="136" OperationName="ComputeAll">
    <ParamList name="">
      <Param macro="SELECTED_START_FRAME" name="START_FRAME"/>
      <Param macro="SELECTED_END_FRAME" name="LAST_FRAME"/>
      <Param macro="ACTIVE_SUBJECTS" name="SUBJECTS"/>
      <Param name="General.FirstProcessingStage" value="1"/>
      <Param name="General.OutputLevel" value="4"/>
      <Param name="General.CalibrationLevel" value="0"/>
      <Param name="General.NumberOfSubjects" value="1"/>
      <Param name="General.FillGaps" value="false"/>
      <Param name="General.ForwardPass" value="true"/>
      <Param name="Input.FilterParameters" value="true"/>
      <Param name="Input.EnableClock" value="false"/>
      <Param name="Input.HaveViconSystem" value="false"/>
      <Param name="Input.Filename" value=""/>
      <Param name="Input.Loop" value="false"/>
      <Param name="Input.LoopCount" value="0"/>
      <Param name="Input.PreambleFrames" value="0"/>
      <Param name="Input.PreamblePeriod" value="0"/>
      <Param name="Input.WarmCache" value="false"/>
      <Param name="Input.XCPFilename" value=""/>
      <Param name="Input.VSKFilenames" value=""/>
      <Param name="Input.CreateDefaultJointRanges" value="false"/>
      <Param name="Input.CreateSubjectSettings" value="false"/>
      <Param name="Input.VSSFilenames" value=""/>
      <Param name="Input.VSSCreateCalibrationTemplates" value="false"/>
      <Param name="Input.VSTFilenames" value=""/>
      <Param name="Input.VideoInputDeviceURNS" value=""/>
      <Param name="Input.DigitalDeviceURNS" value=""/>
      <Param name="Input.DataTransferThreads" value="0"/>
      <Param name="Input.DataProcessingMode" value="5"/>
      <Param name="Input.StartFrame" value="0"/>
      <Param name="Input.EndFrame" value="4294967295"/>
      <Param name="Input.X2DDeriveTickFromTimecode" value="false"/>
      <Param name="Input.X2DAddHardware" value="true"/>
      <Param name="Input.X2DIncludeUserIds" value="false"/>
      <Param name="Input.TypeBlackList" value=""/>
      <Param name="Input.TypeWhiteList" value=""/>
      <Param name="Input.CameraBlackList" value=""/>
      <Param name="Input.CameraWhiteList" value=""/>
      <Param name="Input.SubjectSettingsFilename" value=""/>
      <Param name="Input.MCPStaticItemsFilename" value=""/>
      <Param name="Input.VideoFilenames" value=""/>
      <Param name="Input.VideoFileCameraNames" value=""/>
      <Param name="Input.VideoFileDeviceURNs" value=""/>
      <Param name="Input.VideoFileAddHardware" value="true"/>
      <Param name="Input.ClockUseTimestamp" value="true"/>
      <Param name="Input.ClockFrequency" value="120"/>
      <Param name="Input.HashNames" value="true"/>
      <Param name="Input.RemoteAddress" value=""/>
      <Param name="Input.BlockingStream" value="false"/>
      <Param name="Input.ReadSession" value="0"/>
      <Param name="Input.ZeroFlowStoreOffset" value="false"/>
      <Param name="GlobalItems.Enabled" value="true"/>
      <Param name="GlobalItems.TypeBlackList" value=""/>
      <Param name="GlobalItems.TypeWhiteList" value=""/>
      <Param name="PostGlobalItems.MCPFilename" value=""/>
      <Param name="CircleFitter.ThreadCount" value="0"/>
      <Param name="CircleFitter.Enabled" value="true"/>
      <Param name="CircleFitterType" value="FastSplit"/>
      <Param name="FastFitCircularityThreshold" value="0.25"/>
      <Param name="CentroidSystem.RefitAlreadyFittedBlobs" value="false"/>
      <Param name="PreferCameraSpecificSettings" value="true"/>
      <Param name="MinimumFastSplitWholeBlobRadius" value="1.5"/>
      <Param name="FastSplitPixelBudget" value="1000000"/>
      <Param name="CircularityTweak" value="0"/>
      <Param name="OverrideCircularityThresholdSplit" value="0.44"/>
      <Param name="UseNonGreedyBlobSplitting" value="false"/>
      <Param name="GreedyScoreThreshold" value="0.65000000000000002"/>
      <Param name="CentroidSystem.AllowSinglePixelCentroids" value="false"/>
      <Param name="VideoCentroids.Enabled" value="false"/>
      <Param name="VideoCentroids.PreferCameraSpecificSettings" value="true"/>
      <Param name="VideoCentroids.FastFitCircularityThreshold" value="0.29999999999999999"/>
      <Param name="VideoCentroids.SplitVideoFitters" value="true"/>
      <Param name="VideoCentroids.SubsampleFactor" value="2"/>
      <Param name="Reconstructor.Enabled" value="true"/>
      <Param name="Reconstructor.ThreadCount" value="2"/>
      <Param name="Reconstructor.3DPredictions" value="false"/>
      <Param name="PredictionError" value="150"/>
      <Param name="StartupError" value="150"/>
      <Param name="EnvironmentalDriftTolerance" value="1.5"/>
      <Param name="MinCentroidRadius" value="0"/>
      <Param name="MaxCentroidRadius" value="50"/>
      <Param name="MinReconRadius" value="0"/>
      <Param name="MaxReconRadius" value="1000"/>
      <Param name="MinCams" value="3"/>
      <Param name="MinCamsWithPrediction" value="2"/>
      <Param name="PredictionMatchScoreFactor" value="2"/>
      <Param name="MinSeparation" value="14"/>
      <Param name="MinReconX" value="-100000"/>
      <Param name="MinReconY" value="-100000"/>
      <Param name="MinReconZ" value="-100000"/>
      <Param name="MaxReconX" value="100000"/>
      <Param name="MaxReconY" value="100000"/>
      <Param name="MaxReconZ" value="100000"/>
      <Param name="MatcherLowerMatchLimit" value="0"/>
      <Param name="MatcherUpperMatchLimit" value="0"/>
      <Param name="MatcherAlwaysSort" value="false"/>
      <Param name="CalculateMetrics" value="false"/>
      <Param name="MetricsFilename" value=""/>
      <Param name="RequireLabellingClusters" value="false"/>
      <Param name="Labeller.UseRobustBooting" value="false"/>
      <Param name="BootingOutOfRangeBehaviour" value="FreezeAndUnlabelDownTheChain"/>
      <Param name="Labeller.RigidBodiesMinMatchCount" value="4"/>
      <Param name="EntranceThreshold" value="0.84999999999999998"/>
      <Param name="ExitThreshold" value="0.59999999999999998"/>
      <Param name="BootingQualityHeuristic" value="0"/>
      <Param name="BootingVersusTrackingHeuristic" value="0.5"/>
      <Param name="TrackingQualityHeuristic" value="0"/>
      <Param name="UseUnconstrainessScore" value="true"/>
      <Param name="UnconstrainessEntranceThreshold" value="1"/>
      <Param name="UnconstrainessExitThreshold" value="1.5"/>
      <Param name="TrackingPriorImportance" value="1"/>
      <Param name="TrackingOutOfRangeBehaviour" value="DoNothing"/>
      <Param name="JointRangeThreshold" value="1"/>
      <Param name="TrackingJointRangeThreshold" value="1"/>
      <Param name="Labeller.EnforceRanges" value="false"/>
      <Param name="Labeller.EnableTrackingLine" value="true"/>
      <Param name="BootingKinematicFitThreshold" value="-50"/>
      <Param name="LabelScore" value="1"/>
      <Param name="MergerMarkerImportance" value="0.5"/>
      <Param name="UnassignedReconProb" value="1e-10"/>
      <Param name="UnassignedLabelProb" value="1.0000000000000001e-05"/>
      <Param name="MarkerFitThreshold" value="1.7976931348623157e+308"/>
      <Param name="TrackingKinematicFitThreshold" value="-50"/>
      <Param name="LogMissingMarkerProbability" value="-18.420680743952367"/>
      <Param name="OutOfRangePenalty" value="5"/>
      <Param name="UseFlowFitPool" value="false"/>
      <Param name="IncrementalLabeller" value="false"/>
      <Param name="AllowSubjectReboot" value="true"/>
      <Param name="RigidBodyTranslationSlack" value="100"/>
      <Param name="RigidBodyRotationSlack" value="0.5"/>
      <Param name="Labeller.UnlabelledBootingLines" value="0"/>
      <Param name="GapFiller.UseRevisedTracker" value="true"/>
      <Param name="GapFiller.OverwriteRecons" value="false"/>
      <Param name="GapFiller.TransitionTime" value="0.10000000000000001"/>
      <Param name="SmoothnessHeuristic" value="0"/>
      <Param name="DataFidelityHeuristic" value="0"/>
      <Param name="PriorImportance" value="25"/>
      <Param name="MeanPoseRatio" value="1"/>
      <Param name="SystemUtils.UseSystemHealthReporterV2" value="false"/>
      <Param name="CameraAutoHealBump.Enabled" value="false"/>
      <Param name="SystemUtils.Enabled" value="true"/>
      <Param name="CameraHealing.Enable" value="false"/>
      <Param name="CameraAutoHeal.Enabled" value="false"/>
      <Param name="CameraAutoHeal.XCPFilename" value=""/>
      <Param name="CameraAutoHeal.AutoHealMCPFilename" value=""/>
      <Param name="CameraAutoHeal.AutoStartInGUIMode" value="false"/>
      <Param name="CameraAutoHealBump.MinimumFrameCount" value="250"/>
      <Param name="CameraAutoHealBump.MinOccupiedBuckets" value="12"/>
      <Param name="CameraAccelerometry.Enabled" value="false"/>
      <Param name="GreyscaleCollector.Enabled" value="false"/>
      <Param name="SystemHealth.Enable" value="false"/>
      <Param name="SystemHealthReport.Enabled" value="false"/>
      <Param name="CalibrationReport.Filename" value=""/>
      <Param name="PerformanceMonitor.Enabled" value="false"/>
      <Param name="SystemHealth.WindowSizeInSecondsCalibrationHealth" value="20"/>
      <Param name="SystemHealth.UseLabelledCentroidsOnlyV2" value="false"/>
      <Param name="SystemHealth.StateThrottle" value="1"/>
      <Param name="SystemHealth.AddReconsFromLabelledCentroids" value="false"/>
      <Param name="SystemHealth.AddOnlineMasking" value="false"/>
      <Param name="SystemHealth.OnlineMaskingAddLabelledData" value="false"/>
      <Param name="SystemHealth.OnlineMaskingUpdatePeriodInFrames" value="200"/>
      <Param name="SystemHealth.OnlineMaskingUpdateDurationInFrames" value="10"/>
      <Param name="SystemHealth.IncludeDebugStateCache" value="false"/>
      <Param name="CameraTrackFromCluster.Enabled" value="false"/>
      <Param name="Reconstructor.MaxTrajLengthToDelete" value="0"/>
      <Param name="Output.SubjectWorldPoseGenerator" value="false"/>
      <Param name="Output.CentroidResidualSetGenerator" value="false"/>
      <Param name="Drop.Enabled" value="false"/>
      <Param name="N" value="2"/>
      <Param name="StartOffset" value="0"/>
      <Param name="DropFramesBeforeStart" value="false"/>
      <Param name="DropFramesAfterX" value="-1"/>
      <Param name="Clear" value="true"/>
      <Param name="DropNth" value="false"/>
      <Param name="TargetName" value=""/>
      <Param name="Ouput.AddEventInserter" value="false"/>
      <Param name="Output.EnableConsistencyChecker" value="false"/>
      <Param name="Output.StreamServer" value="false"/>
      <Param name="Output.CGStreamServerEnabled" value="false"/>
      <Param name="Output.C3DFilename" value=""/>
      <Param name="Output.C3DTrajectoryPackingEnabled" value="true"/>
      <Param name="Output.FSSFilename" value=""/>
      <Param name="Output.X2DFilename" value=""/>
      <Param name="Output.X2DWriteBehaviour" value="0"/>
      <Param name="Output.StateCacheName" value="DataStore"/>
      <Param name="Output.StateCache" value="100"/>
      <Param name="Output.StateCacheMB" value="0"/>
      <Param name="Output.StateCacheSeconds" value="0"/>
      <Param name="Output.AddLowLatencyStateCaches" value="false"/>
      <Param name="Output.StateCacheExcluding2D" value="0"/>
      <Param name="Output.StateCacheExcludingVideo" value="0"/>
      <Param name="Output.ServerThreadCount" value="1"/>
      <Param name="Output.ServerPort" value="44603"/>
      <Param name="Output.BlockWhenDisconnected" value="false"/>
      <Param name="Output.TypeBlackList" value=""/>
      <Param name="Output.TypeWhiteList" value=""/>
      <Param name="Output.CameraBlackList" value=""/>
      <Param name="Output.CameraWhiteList" value=""/>
      <Param name="Output.MCPTypeBlackList" value=""/>
      <Param name="Output.MCPTypeWhiteList" value=""/>
      <Param name="Output.CGStreamServerPort" value="8002"/>
      <Param name="Output.CGStreamServerSubjectTypeToStream" value="0"/>
      <Param name="Output.CGStreamServerDataTypeToStream" value="0"/>
      <Param name="General.CacheProgress" value="false"/>
      <Param name="DebugStateCachePattern" value=""/>
      <Param name="DebugStateCacheCapacity" value="1000"/>
      <Param name="DebugFileWriterPattern" value=""/>
      <Param name="DebugFileWriterStem" value=""/>
      <Param name="DebugConsistencyCheckerPattern" value=""/>
      <Param name="DebugFileWriterBlackListNotWhiteList" value="true"/>
      <Param name="DebugFileWriterFilterTypeNames" value=""/>
      <Param name="FitReconVolToTarget" value="false"/>
      <Param name="SINGLE_PASS" value="false"/>
      <Param name="Local.RemoveTemporaryOutput" value="true"/>
      <Param name="Local.Pass1Filename"/>
      <Param name="Local.Pass2Filename"/>
      <Param name="Local.GraphParamsFilename"/>
      <Param name="Local.PostGlobalItemsFilename"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2- Calibration Operation" Enabled="0" OperationId="41" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="NEXUS CGM2.3 Calibration -msm"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="Auto Crop Trial" Enabled="0" OperationId="107" OperationName="AutoCropTrial">
    <ParamList name="">
      <Param macro="FIRST_FRAME" name="FirstFrame"/>
      <Param macro="END_FRAME" name="LastFrame"/>
      <Param name="StartPercent" value="100"/>
      <Param name="EndPercent" value="100"/>
      <Param name="FramesRequired" value="3"/>
      <Param macro="ACTIVE_SUBJECTS" name="SUBJECTS"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="Filter Trajectories - Woltring" Enabled="0" OperationId="54" OperationName="WoltringFilter">
    <ParamList name="">
      <Param macro="SELECTED_START_FRAME" name="FirstFrame"/>
      <Param macro="SELECTED_END_FRAME" name="LastFrame"/>
      <Param name="Mode" value="0"/>
      <Param name="Trajs" value="0"/>
      <Param name="Smoothing" value="20"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-Gaps-Gloersen Operation" Enabled="0" OperationId="108" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="NEXUS Gaps Gloersen"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-Fitting Operation" Enabled="0" OperationId="109" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="NEXUS CGM2.3 Fitting -msm"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-Events-Zeni" Enabled="0" OperationId="110" OperationName="Python">
      <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="NEXUS Events Zeni"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
        <Param name="LaunchPython" value="false"/>
        <Param name="UseNexusPython" value="false"/>
      </ParamList>
    </Entry>


  <Entry DisplayName="pyCGM2-Intell-Events" Enabled="0" OperationId="110" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.path }}pyCGM2\Apps\ViconApps\Events\intelleventDetector.py"/>
      <Param name="ScriptArgs" value=""/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-Plots-Kinematics" Enabled="0" OperationId="0" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="NEXUS Plots Kinematics Normalized"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-Plots-Kinetics" Enabled="0" OperationId="0" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="NEXUS Plots Kinetics Normalized"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="Export C3D" Enabled="0" OperationId="91" OperationName="Exportc3d">
    <ParamList name="">
      <Param macro="CURRENT_TRIAL" name="Filename"/>
      <Param macro="SELECTED_START_FRAME" name="StartFrame"/>
      <Param macro="SELECTED_END_FRAME" name="EndFrame"/>
      <Param name="Postfix" value=""/>
      <Param name="IntegerFormat" value="false"/>
      <Param name="SubjectPrefix" value="1"/>
      <Param name="XAxis" value="4"/>
      <Param name="YAxis" value="2"/>
      <Param name="ZAxis" value="0"/>
      <Param name="ProcessingClip"/>
      <Param name="CopyToClip"/>
    </ParamList>
  </Entry>

</Pipeline>
