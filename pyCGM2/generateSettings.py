# -*- coding: utf-8 -*-
import os
import json
from collections import OrderedDict

CONTENT_INPUTS_CGM1 ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
        "LASI":"",
    	  "RASI":"",
    	  "LPSI":"",
    	  "RPSI":"",
    	  "RTHI":"",
    	  "RKNE":"",
    	  "RTIB":"",
    	  "RANK":"",
          "RMED":"",
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LTIB":"",
    	  "LANK":"",
    	  "LMED":"",
        "LHEE":"",
    	  "LTOE":""
	},
      "Calibration" : {
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "Distal"
      }
    }
    """


CONTENT_INPUTS_CGM1_1 ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
        "LASI":"",
    	  "RASI":"",
    	  "LPSI":"",
    	  "RPSI":"",
    	  "RTHI":"",
    	  "RKNE":"",
    	  "RKNM":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LKNM":"",
        "LTIB":"",
    	  "LANK":"",
    	  "LMED":"",
        "LHEE":"",
    	  "LTOE":""
	},
      "Calibration" : {
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS"
      }
    }
    """

CONTENT_INPUTS_CGM2_1 ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
        "LASI":"",
    	  "RASI":"",
    	  "LPSI":"",
    	  "RPSI":"",
    	  "RTHI":"",
    	  "RKNE":"",
    	  "RKNM":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LKNM":"",
        "LTIB":"",
    	  "LANK":"",
    	  "LMED":"",
        "LHEE":"",
    	  "LTOE":""
	   },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS"
      }
    }
    """

CONTENT_INPUTS_CGM2_2 ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
        "LASI":"",
    	  "RASI":"",
    	  "LPSI":"",
    	  "RPSI":"",
    	  "RTHI":"",
    	  "RKNE":"",
    	  "RKNM":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LKNM":"",
        "LTIB":"",
    	  "LANK":"",
    	  "LMED":"",
        "LHEE":"",
    	  "LTOE":""
	   },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS",
        "Weight" :{
            "LASI" : 100,
            "RASI" : 100,
            "LPSI" : 100,
            "RPSI" : 100,
            "LTHI" : 100,
            "LKNE" : 100,
            "LTIB":  100,
            "LANK" : 100,
            "LHEE" : 100,
            "LTOE" : 100,
            "RTHI" : 100,
            "RKNE" : 100,
            "RTIB":  100,
            "RANK" : 100,
            "RHEE" : 100,
            "RTOE" : 100
        }
      }
    }
    """

CONTENT_INPUTS_CGM2_2_EXPERT ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
        "LASI":"",
    	  "RASI":"",
    	  "LPSI":"",
    	  "RPSI":"",
    	  "RTHI":"",
    	  "RKNE":"",
    	  "RKNM":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LKNM":"",
        "LTIB":"",
    	  "LANK":"",
    	  "LMED":"",
        "LHEE":"",
    	  "LTOE":""
	   },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS",
        "Weight" :{
             "LASI":0,
             "LASI_posAnt":100,
             "LASI_medLat":100,
             "LASI_supInf":100,
             "RASI":0,
             "RASI_posAnt":100,
             "RASI_medLat":100,
             "RASI_supInf":100,
             "LPSI":0,
             "LPSI_posAnt":100,
             "LPSI_medLat":100,
             "LPSI_supInf":100,
             "RPSI":0,
             "RPSI_posAnt":100,
             "RPSI_medLat":100,
             "RPSI_supInf":100,
             "RTHI":0,
             "RTHI_posAnt":100,
             "RTHI_medLat":100,
             "RTHI_proDis":100,
             "RKNE":0,
             "RKNE_posAnt":100,
             "RKNE_medLat":100,
             "RKNE_proDis":100,
             "RTIB":0,
             "RTIB_posAnt":100,
             "RTIB_medLat":100,
             "RTIB_proDis":100,
             "RANK":0,
             "RANK_posAnt":100,
             "RANK_medLat":100,
             "RANK_proDis":100,
             "RHEE":0,
             "RHEE_supInf":100,
             "RHEE_medLat":100,
             "RHEE_proDis":100,
             "RTOE":0,
             "RTOE_supInf":100,
             "RTOE_medLat":100,
             "RTOE_proDis":100,

             "LTHI":0,
             "LTHI_posAnt":100,
             "LTHI_medLat":100,
             "LTHI_proDis":100,
             "LKNE":0,
             "LKNE_posAnt":100,
             "LKNE_medLat":100,
             "LKNE_proDis":100,
             "LTIB":0,
             "LTIB_posAnt":100,
             "LTIB_medLat":100,
             "LTIB_proDis":100,
             "LANK":0,
             "LANK_posAnt":100,
             "LANK_medLat":100,
             "LANK_proDis":100,
             "LHEE":0,
             "LHEE_supInf":100,
             "LHEE_medLat":100,
             "LHEE_proDis":100,
             "LTOE":0,
             "LTOE_supInf":100,
             "LTOE_medLat":100,
             "LTOE_proDis":100
        }
      }
    }
    """

CONTENT_INPUTS_CGM2_3 ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
            "LASI":"",
            "RASI":"",
            "LPSI":"",
            "RPSI":"",
            "RTHI":"",
            "RKNE":"",
            "RKNM":"",
            "RTHAP":"",
            "RTHAD":"",
            "RTIB":"",
            "RANK":"",
            "RMED":"",
            "RTIAP":"",
            "RTIAD":"",
            "RHEE":"",
            "RTOE":"",
            "LTHI":"",
            "LKNE":"",
            "LKNM":"",
            "LTHAP":"",
            "LTHAD":"",
            "LTIB":"",
            "LANK":"",
            "LMED":"",
            "LTIAP":"",
            "LTIAD":"",
            "LHEE":"",
            "LTOE":""
            },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS",
        "Weight" :{
            "LASI":100,
            "RASI":100,
            "LPSI":100,
            "RPSI":100,
            "RTHI":100,
            "RKNE":100,
            "RTHAP":100,
            "RTHAD":100,
            "RTIB":100,
            "RANK":100,
            "RTIAP":100,
            "RTIAD":100,
            "RHEE":100,
            "RTOE":100,
            "LTHI":100,
            "LKNE":100,
            "LTHAP":100,
            "LTHAD":100,
            "LTIB":100,
            "LANK":100,
            "LTIAP":100,
            "LTIAD":100,
            "LHEE":100,
            "LTOE":100,
            "RTHLD":0,
            "RPAT":0,
            "LTHLD":0,
            "LPAT":0
        }
      }
    }
    """

CONTENT_INPUTS_CGM2_3_EXPERT ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
            "LASI":"",
            "RASI":"",
            "LPSI":"",
            "RPSI":"",
            "RTHI":"",
            "RKNE":"",
            "RTHAP":"",
            "RTHAD":"",
            "RTIB":"",
            "RANK":"",
            "RTIAP":"",
            "RTIAD":"",
            "RHEE":"",
            "RTOE":"",
            "LTHI":"",
            "LKNE":"",
            "LTHAP":"",
            "LTHAD":"",
            "LTIB":"",
            "LANK":"",
            "LTIAP":"",
            "LTIAD":"",
            "LHEE":"",
            "LTOE":"",
            "RKNM":"",
            "LKNM":"",
            "RMED":"",
            "LMED":""
            },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS",
        "Weight" :{
            "LASI":0,
             "LASI_posAnt":100,
             "LASI_medLat":100,
             "LASI_supInf":100,
             "RASI":0,
             "RASI_posAnt":100,
             "RASI_medLat":100,
             "RASI_supInf":100,
             "LPSI":0,
             "LPSI_posAnt":100,
             "LPSI_medLat":100,
             "LPSI_supInf":100,
             "RPSI":0,
             "RPSI_posAnt":100,
             "RPSI_medLat":100,
             "RPSI_supInf":100,

             "RTHI":0,
             "RTHI_posAnt":100,
             "RTHI_medLat":100,
             "RTHI_proDis":100,
             "RKNE":0,
             "RKNE_posAnt":100,
             "RKNE_medLat":100,
             "RKNE_proDis":100,
             "RTIB":0,
             "RTIB_posAnt":100,
             "RTIB_medLat":100,
             "RTIB_proDis":100,
             "RANK":0,
             "RANK_posAnt":100,
             "RANK_medLat":100,
             "RANK_proDis":100,
             "RHEE":0,
             "RHEE_supInf":100,
             "RHEE_medLat":100,
             "RHEE_proDis":100,
             "RTOE":0,
             "RTOE_supInf":100,
             "RTOE_medLat":100,
             "RTOE_proDis":100,

             "LTHI":0,
             "LTHI_posAnt":100,
             "LTHI_medLat":100,
             "LTHI_proDis":100,
             "LKNE":0,
             "LKNE_posAnt":100,
             "LKNE_medLat":100,
             "LKNE_proDis":100,
             "LTIB":0,
             "LTIB_posAnt":100,
             "LTIB_medLat":100,
             "LTIB_proDis":100,
             "LANK":0,
             "LANK_posAnt":100,
             "LANK_medLat":100,
             "LANK_proDis":100,
             "LHEE":0,
             "LHEE_supInf":100,
             "LHEE_medLat":100,
             "LHEE_proDis":100,
             "LTOE":0,
             "LTOE_supInf":100,
             "LTOE_medLat":100,
             "LTOE_proDis":100,

             "LTHAP":0,
             "LTHAP_posAnt":100,
             "LTHAP_medLat":100,
             "LTHAP_proDis":100,
             "LTHAD":0,
             "LTHAD_posAnt":100,
             "LTHAD_medLat":100,
             "LTHAD_proDis":100,
             "RTHAP":0,
             "RTHAP_posAnt":100,
             "RTHAP_medLat":100,
             "RTHAP_proDis":100,
             "RTHAD":0,
             "RTHAD_posAnt":100,
             "RTHAD_medLat":100,
             "RTHAD_proDis":100,
             "LTIAP":0,
             "LTIAP_posAnt":100,
             "LTIAP_medLat":100,
             "LTIAP_proDis":100,
             "LTIAD":0,
             "LTIAD_posAnt":100,
             "LTIAD_medLat":100,
             "LTIAD_proDis":100,
             "RTIAP":0,
             "RTIAP_posAnt":100,
             "RTIAP_medLat":100,
             "RTIAP_proDis":100,
             "RTIAD":0,
             "RTIAD_posAnt":100,
             "RTIAD_medLat":100,
             "RTIAD_proDis":100,

             "LTHLD":0,
             "LTHLD_posAnt":0,
             "LTHLD_medLat":0,
             "LTHLD_proDis":0,
             "LPAT":0,
             "LPAT_posAnt":0,
             "LPAT_medLat":0,
             "LPAT_proDis":0,
             "RTHLD":0,
             "RTHLD_posAnt":0,
             "RTHLD_medLat":0,
             "RTHLD_proDis":0,
             "RPAT":0,
             "RPAT_posAnt":0,
             "RPAT_medLat":0,
             "RPAT_proDis":0
        }
      }
    }
    """

CONTENT_INPUTS_CGM2_4 ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
            "LASI":"",
            "RASI":"",
            "LPSI":"",
            "RPSI":"",
            "RTHI":"",
            "RKNE":"",
            "RTIAP":"",
            "RTIAD":"",
            "RTIB":"",
            "RANK":"",
            "RTIAP":"",
            "RTIAD":"",
            "RHEE":"",
            "RSMH":"",
            "RTOE":"",
            "RFMH":"",
            "RVMH":"",
            "LTHI":"",
            "LKNE":"",
            "LTHAP":"",
            "LTHAD":"",
            "LTIB":"",
            "LANK":"",
            "LTIAP":"",
            "LTIAD":"",
            "LHEE":"",
            "LSMH":"",
            "LTOE":"",
            "LFMH":"",
            "LVMH":"",
            "RKNM":"",
            "LKNM":"",
            "RMED":"",
            "LMED":""
            },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS",
        "Weight" :{
            "LASI":100,
            "RASI":100,
            "LPSI":100,
            "RPSI":100,
            "RTHI":100,
            "RKNE":100,
            "RTHAP":100,
            "RTHAD":100,
            "RTIB":100,
            "RANK":100,
            "RTIAP":100,
            "RTIAD":100,
            "RHEE":100,
            "RSMH":0,
            "RTOE":100,
            "RFMH":100,
            "RVMH":100,
            "LTHI":100,
            "LKNE":100,
            "LTHAP":100,
            "LTHAD":100,
            "LTIB":100,
            "LANK":100,
            "LTIAP":100,
            "LTIAD":100,
            "LHEE":100,
            "LSMH":0,
            "LTOE":100,
            "LFMH":100,
            "LVMH":100,
            "RTHLD":0,
            "RPAT":0,
            "LTHLD":0,
            "LPAT": 0
        }
      }
    }
    """


CONTENT_INPUTS_CGM2_4_EXPERT ="""
    {
      "Global" : {
        "Marker diameter" : 14,
        "Point suffix" : ""
      },
      "Translators" : {
            "LASI":"",
            "RASI":"",
            "LPSI":"",
            "RPSI":"",
            "RTHI":"",
            "RKNE":"",
            "RTHAP":"",
            "RTHAD":"",
            "RTIB":"",
            "RANK":"",
            "RTIAP":"",
            "RTIAD":"",
            "RHEE":"",
            "RSMH":"",
            "RTOE":"",
            "RFMH":"",
            "RVMH":"",
            "LTHI":"",
            "LKNE":"",
            "LTHAP":"",
            "LTHAD":"",
            "LTIB":"",
            "LANK":"",
            "LTIAP":"",
            "LTIAD":"",
            "LHEE":"",
            "LSMH":"",
            "LTOE":"",
            "LFMH":"",
            "LVMH":"",
            "RKNM":"",
            "LKNM":"",
            "RMED":"",
            "LMED":""
            },
      "Calibration" : {
        "HJC regression" : "Hara",
        "Left flat foot" : 1 ,
        "Right flat foot" : 1
      },
      "Fitting" : {
        "Moment Projection" : "JCS",
        "Weight" :{
            "LASI":0,
             "LASI_posAnt":100,
             "LASI_medLat":100,
             "LASI_supInf":100,
             "RASI":0,
             "RASI_posAnt":100,
             "RASI_medLat":100,
             "RASI_supInf":100,
             "LPSI":0,
             "LPSI_posAnt":100,
             "LPSI_medLat":100,
             "LPSI_supInf":100,
             "RPSI":0,
             "RPSI_posAnt":100,
             "RPSI_medLat":100,
             "RPSI_supInf":100,

             "RTHI":0,
             "RTHI_posAnt":100,
             "RTHI_medLat":100,
             "RTHI_proDis":100,
             "RKNE":0,
             "RKNE_posAnt":100,
             "RKNE_medLat":100,
             "RKNE_proDis":100,
             "RTIB":0,
             "RTIB_posAnt":100,
             "RTIB_medLat":100,
             "RTIB_proDis":100,
             "RANK":0,
             "RANK_posAnt":100,
             "RANK_medLat":100,
             "RANK_proDis":100,
             "RHEE":0,
             "RHEE_supInf":100,
             "RHEE_medLat":100,
             "RHEE_proDis":100,
             "RSMH":0,
             "RSMH_supInf":0,
             "RSMH_medLat":0,
             "RSMH_proDis":0,
             "RTOE":0,
             "RTOE_supInf":100,
             "RTOE_medLat":100,
             "RTOE_proDis":100,
             "RFMH":0,
             "RFMH_supInf":100,
             "RFMH_medLat":100,
             "RFMH_proDis":100,
             "RVMH":0,
             "RVMH_supInf":100,
             "RVMH_medLat":100,
             "RVMH_proDis":100,

             "LTHI":0,
             "LTHI_posAnt":100,
             "LTHI_medLat":100,
             "LTHI_proDis":100,
             "LKNE":0,
             "LKNE_posAnt":100,
             "LKNE_medLat":100,
             "LKNE_proDis":100,
             "LTIB":0,
             "LTIB_posAnt":100,
             "LTIB_medLat":100,
             "LTIB_proDis":100,
             "LANK":0,
             "LANK_posAnt":100,
             "LANK_medLat":100,
             "LANK_proDis":100,
             "LHEE":0,
             "LHEE_supInf":100,
             "LHEE_medLat":100,
             "LHEE_proDis":100,
             "LSMH":0,
             "LSMH_supInf":0,
             "LSMH_medLat":0,
             "LSMH_proDis":0,
             "LTOE":0,
             "LTOE_supInf":100,
             "LTOE_medLat":100,
             "LTOE_proDis":100,
             "LFMH":0,
             "LFMH_supInf":100,
             "LFMH_medLat":100,
             "LFMH_proDis":100,
             "LVMH":0,
             "LVMH_supInf":100,
             "LVMH_medLat":100,
             "LVMH_proDis":100,

             "LTHAP":0,
             "LTHAP_posAnt":100,
             "LTHAP_medLat":100,
             "LTHAP_proDis":100,
             "LTHAD":0,
             "LTHAD_posAnt":100,
             "LTHAD_medLat":100,
             "LTHAD_proDis":100,
             "RTHAP":0,
             "RTHAP_posAnt":100,
             "RTHAP_medLat":100,
             "RTHAP_proDis":100,
             "RTHAD":0,
             "RTHAD_posAnt":100,
             "RTHAD_medLat":100,
             "RTHAD_proDis":100,
             "LTIAP":0,
             "LTIAP_posAnt":100,
             "LTIAP_medLat":100,
             "LTIAP_proDis":100,
             "LTIAD":0,
             "LTIAD_posAnt":100,
             "LTIAD_medLat":100,
             "LTIAD_proDis":100,
             "RTIAP":0,
             "RTIAP_posAnt":100,
             "RTIAP_medLat":100,
             "RTIAP_proDis":100,
             "RTIAD":0,
             "RTIAD_posAnt":100,
             "RTIAD_medLat":100,
             "RTIAD_proDis":100,

             "LTHLD":0,
             "LTHLD_posAnt":0,
             "LTHLD_medLat":0,
             "LTHLD_proDis":0,
             "LPAT":0,
             "LPAT_posAnt":0,
             "LPAT_medLat":0,
             "LPAT_proDis":0,
             "RTHLD":0,
             "RTHLD_posAnt":0,
             "RTHLD_medLat":0,
             "RTHLD_proDis":0,
             "RPAT":0,
             "RPAT_posAnt":0,
             "RPAT_medLat":0,
             "RPAT_proDis":0
        }
      }
    }
    """


def generateCGM1_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM1-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM1,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM1-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM1_1_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM1_1-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM1_1,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM1_1-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM2_1_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_1-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM2_1,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM2_1-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM2_2_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_2-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM2_2,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM2_2-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM2_2_Expert_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_2-Expert-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM2_2_EXPERT,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM2_2-Expert-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM2_3_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_3-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM2_3,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM2_3-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM2_3_Expert_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_3-Expert-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM2_3_EXPERT,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM2_3-Expert-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()


def generateCGM2_4_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_4-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM2_4,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM2_4-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()

def generateCGM2_4_Expert_Settings(userAppData_path):

    if not os.path.isfile( userAppData_path + "CGM2_4-Expert-pyCGM2.settings"):
        inputs = json.loads(CONTENT_INPUTS_CGM2_4_EXPERT,object_pairs_hook=OrderedDict)

        F = open(str(userAppData_path+"CGM2_4-Expert-pyCGM2.settings"),"w")
        F.write( json.dumps(inputs, sort_keys=False,indent=2, separators=(',', ': ')))
        F.close()
