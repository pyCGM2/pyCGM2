Global:
    {%- if data["ModelVersion"] == "CGM1.1" %} 
    ModelVersion: CGM1.1
    {% elif data["ModelVersion"] == "CGM2.1" %} 
    ModelVersion: CGM2.1
    {% elif data["ModelVersion"] == "CGM2.2" %} 
    ModelVersion: CGM2.2
    {% elif data["ModelVersion"] == "CGM2.3" %} 
    ModelVersion: CGM2.3
    {% elif data["ModelVersion"] == "CGM2.4" %} 
    ModelVersion: CGM2.4
    {% elif data["ModelVersion"] == "CGM2.5" %} 
    ModelVersion: CGM2.5
    {%- else %} 
    ModelVersion: CGM1.0
    {%- endif %}
    MarkerDiameter: 14
    PointSuffix:
    {%- if data["ModelVersion"] != "CGM1.0" %}
    Moment Projection: JSC #[string](choice: Proximal, Global, Distal)
    {%- else %}
    Moment Projection: Distal #[string](choice: Proximal, Global, Distal)
    {%- endif %}
    {%- if data["ModelVersion"] in ["CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}
    EnableIK: 1
    IkAccuracy: 1e-8
    {%- endif %}



#-------------------------------------------------------------------------------
SubjectInfo:
  Ipp: "{{ data["Patient"]["PatientID"] }}"
  Name:
  FirstName:
  Dob:
  InjuredLeg:

#-------------------------------------------------------------------------------
VisitInfo:
  Date: {{ data["Visit"]["Date"] }}
  SessionNumber: {{ data["Visit"]["SessionID"] }}
  Age: {{ data["Visit"]["Age"] }}
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
ExamInfo:
  Goal :
  Comments:
#-------------------------------------------------------------------------------
MP:
    Required:
        Bodymass: {{ data["Mp"]["Bodymass"] }}
        Height: {{ data["Mp"]["Height"] }}
        LeftLegLength: {{ data["Mp"]["LeftLegLength"] }}
        RightLegLength: {{ data["Mp"]["RightLegLength"] }}
        LeftKneeWidth: {{ data["Mp"]["LeftKneeWidth"] }}
        RightKneeWidth: {{ data["Mp"]["RightKneeWidth"] }}
        LeftAnkleWidth: {{ data["Mp"]["LeftAnkleWidth"] }}
        RightAnkleWidth: {{ data["Mp"]["RightAnkleWidth"] }}
        LeftSoleDelta: {{ data["Mp"]["LeftSoleDelta"] }}
        RightSoleDelta: {{ data["Mp"]["RightSoleDelta"] }}
        LeftShoulderOffset: {{ data["Mp"]["LeftShoulderOffset"] }}
        LeftElbowWidth: {{ data["Mp"]["LeftElbowWidth"] }}
        LeftWristWidth: {{ data["Mp"]["LeftWristWidth"] }}
        LeftHandThickness: {{ data["Mp"]["LeftHandThickness"] }}
        RightShoulderOffset: {{ data["Mp"]["RightShoulderOffset"] }}
        RightElbowWidth: {{ data["Mp"]["RightElbowWidth"] }}
        RightWristWidth: {{ data["Mp"]["RightWristWidth"] }}
        RightHandThickness: {{ data["Mp"]["RightHandThickness"] }}
    Optional:
        InterAsisDistance: 0
        LeftAsisTrocanterDistance: 0
        LeftTibialTorsion: 0
        LeftThighRotation: 0
        LeftShankRotation: 0
        RightAsisTrocanterDistance: 0
        RightTibialTorsion: 0
        RightThighRotation: 0
        RightShankRotation: 0
        LeftKneeFuncCalibrationOffset: 0
        RightKneeFuncCalibrationOffset: 0



Calibration:
  {%- if data.Calibration %}
  {% for it in data['Calibration'] %}
  - ID: {{ it[0] }}
    StaticTrial: {{ it[1] }}
    LeftFlatFoot: {{ it[2] }}
    RightFlatFoot: {{ it[3] }}
    HeadFlat: True
    Translators:
        LASI:   
        RASI: 
        LPSI: 
        RPSI: 
        RTHI: 
        RKNE: 
        RTIB: 
        RANK: 
        RMED: 
        RHEE: 
        RTOE: 
        LTHI: 
        LKNE: 
        LTIB: 
        LANK:
        LMED: 
        LHEE: 
        LTOE: 
        {%- if data["ModelVersion"] == "CGM2.5" %}
        C7: T2 # cf Armand et al.2013 https://doi.org/10.1016/j.gaitpost.2013.06.016
        {%- else %}
        C7:
        {%- endif %}
        T10: 
        CLAV:
        {%- if data["ModelVersion"] == "CGM2.5" %} 
        STRN: CLAV
        LFHD: GLAB #Eames et al.1999 10.1016/S0167-9457(99)00022-6
        LBHD: LMAS
        RFHD: GLAB
        RBHD: RMAS
        {%- else %}
        STRN: 
        LFHD: 
        LBHD: 
        RFHD: 
        RBHD: 
        {%- endif %}
        LSHO: 
        LELB: 
        LWRB: 
        LWRA: 
        LFIN: 
        RSHO: 
        RELB: 
        RWRB: 
        RWRA: 
        RFIN: 
        {%- if data["ModelVersion"] in ["CGM1.1","CGM2.1", "CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}
        RKNM: 
        LKNM:
        {%- endif %}
        {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %}
        RTHAP: 
        RTHAD: 
        RTIAP: 
        RTIAD: 
        LTHAP: 
        LTHAD:
        LTIAP: 
        LTIAD:
        {%- endif %}
    {%- if data["ModelVersion"] in ["CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}    
    Weight:
        LASI: 100 #[double]: marker weight used by the Inverse Kinematics solver
        RASI: 100
        LPSI: 100
        RPSI: 100
        RTHI: 100
        RKNE: 100
        RTIB: 100
        RANK: 100
        RHEE: 100
        RTOE: 100
        LTHI: 100
        LKNE: 100
        LTIB: 100
        LANK: 100
        LHEE: 100
        LTOE: 100
        {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %} 
        RTHAP: 100
        RTHAD: 100
        RTIAP: 100
        RTIAD: 100
        LTHAP: 100
        LTHAD: 100
        LTIAP: 100
        LTIAD: 100
        {%- endif %}
        {%- if data["ModelVersion"] in [ "CGM2.4","CGM2.5"] %} 
        RSMH: 0
        RFMH: 100
        RVMH: 100
        LSMH: 0
        LFMH: 100
        LVMH: 100
        {%- endif %}
        #RTHLD: 0
        #RPAT: 0
        #LTHLD: 0
        #LPAT: 0
    {%- endif %}    
  {% endfor %}
  {%- else %}
  - ID: 
    StaticTrial: 
    LeftFlatFoot: 
    RightFlatFoot: 
    HeadFlat: True
    Translators:
        LASI:   
        RASI: 
        LPSI: 
        RPSI: 
        RTHI: 
        RKNE: 
        RTIB: 
        RANK: 
        RMED: 
        RHEE: 
        RTOE: 
        LTHI: 
        LKNE: 
        LTIB: 
        LANK:
        LMED: 
        LHEE: 
        LTOE: 
        {%- if data["ModelVersion"] == "CGM2.5" %}
        C7: T2 # cf Armand et al.2013 https://doi.org/10.1016/j.gaitpost.2013.06.016
        {%- else %}
        C7:
        {%- endif %}
        T10: 
        CLAV:
        {%- if data["ModelVersion"] == "CGM2.5" %} 
        STRN: CLAV
        LFHD: GLAB #Eames et al.1999 10.1016/S0167-9457(99)00022-6
        LBHD: LMAS
        RFHD: GLAB
        RBHD: RMAS
        {%- else %}
        STRN: 
        LFHD: 
        LBHD: 
        RFHD: 
        RBHD: 
        {%- endif %}
        LSHO: 
        LELB: 
        LWRB: 
        LWRA: 
        LFIN: 
        RSHO: 
        RELB: 
        RWRB: 
        RWRA: 
        RFIN: 
        {%- if data["ModelVersion"] in ["CGM1.1","CGM2.1", "CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}
        RKNM: 
        LKNM:
        {%- endif %}
        {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %}
        RTHAP: 
        RTHAD: 
        RTIAP: 
        RTIAD: 
        LTHAP: 
        LTHAD:
        LTIAP: 
        LTIAD:
        {%- endif %}
    {%- if data["ModelVersion"] in ["CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}    
    Weight:
        LASI: 100 #[double]: marker weight used by the Inverse Kinematics solver
        RASI: 100
        LPSI: 100
        RPSI: 100
        RTHI: 100
        RKNE: 100
        RTIB: 100
        RANK: 100
        RHEE: 100
        RTOE: 100
        LTHI: 100
        LKNE: 100
        LTIB: 100
        LANK: 100
        LHEE: 100
        LTOE: 100
        {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %} 
        RTHAP: 100
        RTHAD: 100
        RTIAP: 100
        RTIAD: 100
        LTHAP: 100
        LTHAD: 100
        LTIAP: 100
        LTIAD: 100
        {%- endif %}
        {%- if data["ModelVersion"] in [ "CGM2.4","CGM2.5"] %} 
        RSMH: 0
        RFMH: 100
        RVMH: 100
        LSMH: 0
        LFMH: 100
        LVMH: 100
        {%- endif %}
        #RTHLD: 0
        #RPAT: 0
        #LTHLD: 0
        #LPAT: 0
    {%- endif %}   

  {%- endif %}

Fitting:
    Trials:
      {%- if data.Fitting %}
      {% for it in data['Fitting'] %}
      - File: {{ it[0] }}
        CalibrationID: {{ it[1] }}
        Mfpa: {{ it[2] }}
        ConditionId: {{ it[3] }}
        Emg: True
        Translators:
        LASI:   
        RASI: 
        LPSI: 
        RPSI: 
        RTHI: 
        RKNE: 
        RTIB: 
        RANK: 
        RMED: 
        RHEE: 
        RTOE: 
        LTHI: 
        LKNE: 
        LTIB: 
        LANK:
        LMED: 
        LHEE: 
        LTOE: 
        {%- if data["ModelVersion"] == "CGM2.5" %}
        C7: T2 # cf Armand et al.2013 https://doi.org/10.1016/j.gaitpost.2013.06.016
        {%- else %}
        C7:
        {%- endif %}
        T10: 
        CLAV:
        {%- if data["ModelVersion"] == "CGM2.5" %} 
        STRN: CLAV
        LFHD: GLAB #Eames et al.1999 10.1016/S0167-9457(99)00022-6
        LBHD: LMAS
        RFHD: GLAB
        RBHD: RMAS
        {%- else %}
        STRN: 
        LFHD: 
        LBHD: 
        RFHD: 
        RBHD: 
        {%- endif %}
        LSHO: 
        LELB: 
        LWRB: 
        LWRA: 
        LFIN: 
        RSHO: 
        RELB: 
        RWRB: 
        RWRA: 
        RFIN: 
        {%- if data["ModelVersion"] in ["CGM1.1","CGM2.1", "CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}
        RKNM: 
        LKNM:
        {%- endif %}
        {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %}
        RTHAP: 
        RTHAD: 
        RTIAP: 
        RTIAD: 
        LTHAP: 
        LTHAD:
        LTIAP: 
        LTIAD:
        {%- endif %}
        {%- if data["ModelVersion"] in ["CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}    
        Weight:
            LASI: 100 #[double]: marker weight used by the Inverse Kinematics solver
            RASI: 100
            LPSI: 100
            RPSI: 100
            RTHI: 100
            RKNE: 100
            RTIB: 100
            RANK: 100
            RHEE: 100
            RTOE: 100
            LTHI: 100
            LKNE: 100
            LTIB: 100
            LANK: 100
            LHEE: 100
            LTOE: 100
            {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %} 
            RTHAP: 100
            RTHAD: 100
            RTIAP: 100
            RTIAD: 100
            LTHAP: 100
            LTHAD: 100
            LTIAP: 100
            LTIAD: 100
            {%- endif %}
            {%- if data["ModelVersion"] in [ "CGM2.4","CGM2.5"] %} 
            RSMH: 0
            RFMH: 100
            RVMH: 100
            LSMH: 0
            LFMH: 100
            LVMH: 100
            {%- endif %}
            #RTHLD: 0
            #RPAT: 0
            #LTHLD: 0
            #LPAT: 0
        {%- endif %}
      {% endfor %}
      {%- else %}
      - File: 
        CalibrationID: 
        Mfpa: 
        ConditionId: 
        Emg: True
        Translators:
        LASI:   
        RASI: 
        LPSI: 
        RPSI: 
        RTHI: 
        RKNE: 
        RTIB: 
        RANK: 
        RMED: 
        RHEE: 
        RTOE: 
        LTHI: 
        LKNE: 
        LTIB: 
        LANK:
        LMED: 
        LHEE: 
        LTOE: 
        {%- if data["ModelVersion"] == "CGM2.5" %}
        C7: T2 # cf Armand et al.2013 https://doi.org/10.1016/j.gaitpost.2013.06.016
        {%- else %}
        C7:
        {%- endif %}
        T10: 
        CLAV:
        {%- if data["ModelVersion"] == "CGM2.5" %} 
        STRN: CLAV
        LFHD: GLAB #Eames et al.1999 10.1016/S0167-9457(99)00022-6
        LBHD: LMAS
        RFHD: GLAB
        RBHD: RMAS
        {%- else %}
        STRN: 
        LFHD: 
        LBHD: 
        RFHD: 
        RBHD: 
        {%- endif %}
        LSHO: 
        LELB: 
        LWRB: 
        LWRA: 
        LFIN: 
        RSHO: 
        RELB: 
        RWRB: 
        RWRA: 
        RFIN: 
        {%- if data["ModelVersion"] in ["CGM1.1","CGM2.1", "CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}
        RKNM: 
        LKNM:
        {%- endif %}
        {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %}
        RTHAP: 
        RTHAD: 
        RTIAP: 
        RTIAD: 
        LTHAP: 
        LTHAD:
        LTIAP: 
        LTIAD:
        {%- endif %}
        {%- if data["ModelVersion"] in ["CGM2.2", "CGM2.3", "CGM2.4","CGM2.5"] %}    
        Weight:
            LASI: 100 #[double]: marker weight used by the Inverse Kinematics solver
            RASI: 100
            LPSI: 100
            RPSI: 100
            RTHI: 100
            RKNE: 100
            RTIB: 100
            RANK: 100
            RHEE: 100
            RTOE: 100
            LTHI: 100
            LKNE: 100
            LTIB: 100
            LANK: 100
            LHEE: 100
            LTOE: 100
            {%- if data["ModelVersion"] in ["CGM2.3", "CGM2.4","CGM2.5"] %} 
            RTHAP: 100
            RTHAD: 100
            RTIAP: 100
            RTIAD: 100
            LTHAP: 100
            LTHAD: 100
            LTIAP: 100
            LTIAD: 100
            {%- endif %}
            {%- if data["ModelVersion"] in [ "CGM2.4","CGM2.5"] %} 
            RSMH: 0
            RFMH: 100
            RVMH: 100
            LSMH: 0
            LFMH: 100
            LVMH: 100
            {%- endif %}
            #RTHLD: 0
            #RPAT: 0
            #LTHLD: 0
            #LPAT: 0
        {%- endif %}
    {%- endif %}

Emg:
  Trials:
    {%- if data.Emg %}
    {% for it in data['Emg'] %}
    - File: {{ it[0] }}
      ConditionId: {{ it[1] }}
    {% endfor %}
    {%- else %}
    - File: 
      ConditionId: 
    {%- endif %}  


#-------------------------------------------------------------------------------
Protocol:
  ResearchProtocol:

  Conditions:
    {%- if data.Conditions %}
    {% for it in data['Conditions'] %}
    - ConditionID : {{ it["ConditionID"] }}
      Context : {{ it["Context"] }}
      ContextComments:
      NerveBlock : {{ it["Block"] }}
      Task: {{ it["Task"] }}
      Shoes: {{ it["Shoes"] }}
      Orthosis: {{ it["ProthesisOrthosis"] }}
      ExternalAid: {{ it["ExternalAid"] }}
      PersonalAid: {{ it["PersonalAid"] }}
      Comments:
      Assessor:
      EmgReferenceConditionID: {{ it["EmgReferenceConditionID"] }}
      EmgRepresentativeTrial: {{ it["EmgRepresentativeTrial"] }}
      #----example of overload -----
      EmgSettings:
        CHANNELS:
          Voltage.EMG1 :
              Muscle : RECFEM #[string]
              Context : Left #[string](choice: Left or Right)
              NormalActivity : RECFEM #[string](choice: see above)

          Voltage.EMG2 :
              Muscle : RECFEM
              Context : Right
              NormalActivity : RECFEM

          Voltage.EMG3 :
              Muscle : VASLAT
              Context : Left
              NormalActivity : VASLAT

          Voltage.EMG4 :
              Muscle : VASLAT
              Context : Right
              NormalActivity : VASLAT

          Voltage.EMG5 :
              Muscle : SEMITE
              Context : Left
              NormalActivity : SEMITE

          Voltage.EMG6 :
              Muscle : SEMITE
              Context : Right
              NormalActivity : SEMITE

          Voltage.EMG7 :
              Muscle : TIBANT
              Context : Left
              NormalActivity : TIBANT

          Voltage.EMG8 :
              Muscle : TIBANT
              Context : Right
              NormalActivity : TIBANT

          Voltage.EMG9 :
              Muscle : SOLEUS
              Context : Left
              NormalActivity : SOLEUS

          Voltage.EMG10 :
              Muscle : SOLEUS
              Context : Right
              NormalActivity : SOLEUS

          Voltage.EMG11 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG12 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG13 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG14 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG15 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG16 :
              Muscle :
              Context :
              NormalActivity : 
        Processing:
          BandpassFrequencies: [20,400]
          EnvelopLowpassFrequency: 6
    {% endfor %}
    {%- else %}
    - ConditionID : 
      Context : 
      ContextComments:
      NerveBlock : 
      Task: 
      Shoes: 
      Orthosis: 
      ExternalAid: 
      PersonalAid: 
      Comments:
      Assessor:
      EmgReferenceConditionID: 
      EmgRepresentativeTrial: 
      #----example of overload -----
      EmgSettings:
        CHANNELS:
          Voltage.EMG1 :
              Muscle : RECFEM #[string]
              Context : Left #[string](choice: Left or Right)
              NormalActivity : RECFEM #[string](choice: see above)

          Voltage.EMG2 :
              Muscle : RECFEM
              Context : Right
              NormalActivity : RECFEM

          Voltage.EMG3 :
              Muscle : VASLAT
              Context : Left
              NormalActivity : VASLAT

          Voltage.EMG4 :
              Muscle : VASLAT
              Context : Right
              NormalActivity : VASLAT

          Voltage.EMG5 :
              Muscle : SEMITE
              Context : Left
              NormalActivity : SEMITE

          Voltage.EMG6 :
              Muscle : SEMITE
              Context : Right
              NormalActivity : SEMITE

          Voltage.EMG7 :
              Muscle : TIBANT
              Context : Left
              NormalActivity : TIBANT

          Voltage.EMG8 :
              Muscle : TIBANT
              Context : Right
              NormalActivity : TIBANT

          Voltage.EMG9 :
              Muscle : SOLEUS
              Context : Left
              NormalActivity : SOLEUS

          Voltage.EMG10 :
              Muscle : SOLEUS
              Context : Right
              NormalActivity : SOLEUS

          Voltage.EMG11 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG12 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG13 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG14 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG15 :
              Muscle :
              Context :
              NormalActivity :

          Voltage.EMG16 :
              Muscle :
              Context :
              NormalActivity : 
        Processing:
          BandpassFrequencies: [20,400]
          EnvelopLowpassFrequency: 6
    {%- endif %}

#-------------------------------------------------------------------------------
########## EMG configuration ##########
# The configuration below considers the emg signals are named EMG1,EMG2... EMG16 in your c3d.
# items:
#  - "Muscle" is the name of the muscle where is placed the emg device
#  - "Context" indicates if you want to plot on a left or right gait cycle.
#  - "NormalActivity" defines the muscle whose normal activity will be plotted in background.
#     Values are the normal adult activities defined in Vicon Clinical Manager.
#     you can select :
#         -ADDBRE
#         -ADDLON
#         -ADDMAG
#         -BICFEM
#         -CALF
#         -EXTDIGLON
#         -EXTHALLON
#         -FLEDIGLON
#         -FLEHALLON
#         -GASTRO
#         -GLUMAX
#         -GLUMED
#         -GLUMIN
#         -GRACIL
#         -HAMSTR
#         -HIPABD
#         -HIPADD
#         -HIPEXT
#         -HIPFLE
#         -ILIACU
#         -ILIOPS
#         -LATHAM
#         -LATQUA
#         -MEDHAM
#         -MEDQUA
#         -PERBRE
#         -PERLON
#         -POPLIT
#         -RECFEM
#         -SARTOR
#         -SEMIME
#         -SEMITE
#         -SOLEUS
#         -TENFACLAT
#         -TIBANT
#         -TIBPOS
#         -VASINT
#         -VASLAT
#         -VASMEDLON
#         -VASMEDOBL
########## EMG configuration ##########