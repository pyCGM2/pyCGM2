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
    	  "RMEPI":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",  
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LMEPI":"",
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
    	  "RMEPI":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",  
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LMEPI":"",
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
    	  "RMEPI":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",  
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LMEPI":"",
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
    	  "RMEPI":"",
        "RTIB":"",
    	  "RANK":"",
        "RMED":"",  
    	  "RHEE":"",
    	  "RTOE":"",
    	  "LTHI":"",
    	  "LKNE":"",
    	  "LMEPI":"",
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
            "RMEPI":"",
            "RTHIAP":"",
            "RTHIAD":"",
            "RTIB":"",
            "RANK":"",
            "RMED":"",
            "RTIBAP":"",
            "RTIBAD":"",
            "RHEE":"",
            "RTOE":"",
            "LTHI":"",
            "LKNE":"",
            "LMEPI":"",
            "LTHIAP":"",
            "LTHIAD":"",
            "LTIB":"",
            "LANK":"",
            "LMED":"",
            "LTIBAP":"",
            "LTIBAD":"",
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
            "RTHIAP":100,
            "RTHIAD":100,
            "RTIB":100,
            "RANK":100,
            "RTIBAP":100,
            "RTIBAD":100,
            "RHEE":100,
            "RTOE":100,
            "LTHI":100,
            "LKNE":100,
            "LTHIAP":100,
            "LTHIAD":100,
            "LTIB":100,
            "LANK":100,
            "LTIBAP":100,
            "LTIBAD":100,
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
            "RTHIAP":"",
            "RTHIAD":"",
            "RTIB":"",
            "RANK":"",
            "RTIBAP":"",
            "RTIBAD":"",
            "RHEE":"",
            "RTOE":"",
            "LTHI":"",
            "LKNE":"",
            "LTHIAP":"",
            "LTHIAD":"",
            "LTIB":"",
            "LANK":"",
            "LTIBAP":"",
            "LTIBAD":"",
            "LHEE":"",
            "LTOE":"",
            "RMEPI":"",
            "LMEPI":"",
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
             
             "LTHIAP":0,
             "LTHIAP_posAnt":100,
             "LTHIAP_medLat":100,
             "LTHIAP_proDis":100,                 
             "LTHIAD":0,
             "LTHIAD_posAnt":100,
             "LTHIAD_medLat":100,
             "LTHIAD_proDis":100,
             "RTHIAP":0,
             "RTHIAP_posAnt":100,
             "RTHIAP_medLat":100,
             "RTHIAP_proDis":100,                 
             "RTHIAD":0,
             "RTHIAD_posAnt":100,
             "RTHIAD_medLat":100,
             "RTHIAD_proDis":100,
             "LTIBAP":0,
             "LTIBAP_posAnt":100,
             "LTIBAP_medLat":100,
             "LTIBAP_proDis":100,                 
             "LTIBAD":0,
             "LTIBAD_posAnt":100,
             "LTIBAD_medLat":100,
             "LTIBAD_proDis":100,
             "RTIBAP":0,
             "RTIBAP_posAnt":100,
             "RTIBAP_medLat":100,
             "RTIBAP_proDis":100,                 
             "RTIBAD":0,
             "RTIBAD_posAnt":100,
             "RTIBAD_medLat":100,
             "RTIBAD_proDis":100,
             
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
            "RTHIAP":"",
            "RTHIAD":"",
            "RTIB":"",
            "RANK":"",
            "RTIBAP":"",
            "RTIBAD":"",
            "RHEE":"",
            "RTOE":"",
            "RCUN":"",
            "RD1M":"",
            "RD5M":"",
            "LTHI":"",
            "LKNE":"",
            "LTHIAP":"",
            "LTHIAD":"",
            "LTIB":"",
            "LANK":"",
            "LTIBAP":"",
            "LTIBAD":"",
            "LHEE":"",
            "LTOE":"",
            "LCUN":"",
            "LD1M":"",
            "LD5M":"",
            "RMEPI":"",
            "LMEPI":"",
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
            "RTHIAP":100,
            "RTHIAD":100,
            "RTIB":100,
            "RANK":100,
            "RTIBAP":100,
            "RTIBAD":100,
            "RHEE":100,
            "RTOE":100,
            "RCUN":100,
            "RD1M":100,
            "RD5M":100,
            "LTHI":100,
            "LKNE":100,
            "LTHIAP":100,
            "LTHIAD":100,
            "LTIB":100,
            "LANK":100,
            "LTIBAP":100,
            "LTIBAD":100,
            "LHEE":100,
            "LTOE":100,
            "LCUN":100,
            "LD1M":100,
            "LD5M":100,
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
            "RTHIAP":"",
            "RTHIAD":"",
            "RTIB":"",
            "RANK":"",
            "RTIBAP":"",
            "RTIBAD":"",
            "RHEE":"",
            "RTOE":"",
            "RCUN":"",
            "RD1M":"",
            "RD5M":"",
            "LTHI":"",
            "LKNE":"",
            "LTHIAP":"",
            "LTHIAD":"",
            "LTIB":"",
            "LANK":"",
            "LTIBAP":"",
            "LTIBAD":"",
            "LHEE":"",
            "LTOE":"",
            "LCUN":"",
            "LD1M":"",
            "LD5M":"",
            "RMEPI":"",
            "LMEPI":"",
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
             "RCUN":0,
             "RCUN_supInf":100,
             "RCUN_medLat":100,
             "RCUN_proDis":100,
             "RD1M":0,
             "RD1M_supInf":100,
             "RD1M_medLat":100,
             "RD1M_proDis":100,
             "RD5M":0,
             "RD5M_supInf":100,
             "RD5M_medLat":100,
             "RD5M_proDis":100,

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
             "LCUN":0,
             "LCUN_supInf":100,
             "LCUN_medLat":100,
             "LCUN_proDis":100,
             "LD1M":0,
             "LD1M_supInf":100,
             "LD1M_medLat":100,
             "LD1M_proDis":100,
             "LD5M":0,
             "LD5M_supInf":100,
             "LD5M_medLat":100,
             "LD5M_proDis":100,
             
             "LTHIAP":0,
             "LTHIAP_posAnt":100,
             "LTHIAP_medLat":100,
             "LTHIAP_proDis":100,                 
             "LTHIAD":0,
             "LTHIAD_posAnt":100,
             "LTHIAD_medLat":100,
             "LTHIAD_proDis":100,
             "RTHIAP":0,
             "RTHIAP_posAnt":100,
             "RTHIAP_medLat":100,
             "RTHIAP_proDis":100,                 
             "RTHIAD":0,
             "RTHIAD_posAnt":100,
             "RTHIAD_medLat":100,
             "RTHIAD_proDis":100,
             "LTIBAP":0,
             "LTIBAP_posAnt":100,
             "LTIBAP_medLat":100,
             "LTIBAP_proDis":100,                 
             "LTIBAD":0,
             "LTIBAD_posAnt":100,
             "LTIBAD_medLat":100,
             "LTIBAD_proDis":100,
             "RTIBAP":0,
             "RTIBAP_posAnt":100,
             "RTIBAP_medLat":100,
             "RTIBAP_proDis":100,                 
             "RTIBAD":0,
             "RTIBAD_posAnt":100,
             "RTIBAD_medLat":100,
             "RTIBAD_proDis":100,
             
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