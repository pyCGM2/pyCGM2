import pyCGM2
LOGGER = pyCGM2.LOGGER

import pandas as pd

# Gender	Age (min)	Age (max)	Pace	
# Velocity (m.sec)_min	Velocity (m.sec)_max   ===> speed
# 	Cadence (steps.min)_min	Cadence (steps.min)_max ===> cadence
# 	Stride Time (sec)_min	Stride Time (sec)_max ===> duration
# 	Step Time (sec)_min	Step Time (sec)_max ===> stepDuration
# 	Stride Length (m)_min	Stride Length (m)_max   ===> strideLength
# 	Step Length (m)_min	Step Length (m)_max   ===> stepLength
# 	Stride Width (m)_min	Stride Width (m)_max ===> strideWidth
# 	Swing %_min	Swing %_max	====> swingPhase
# Stance %_min	Stance %_max	====> stancePhase
# Single Support %_min	# Single Support %_max	====> simpleStance
# Total D. Support %_min 	Total D. Support %_max   ====> doubleStance1 and doubleStance2 
# 	GVI_min	GVI_max	SS COP Path Eff. %_min	SS COP Path Eff. %_max	Walk Ratio (m.(steps.min))_min	Walk Ratio (m.(steps.min))_max



#list of all spatio temporal parameters :
            #['Lcadence', 'LdoubleStance1', 'LdoubleStance1Duration', 'LdoubleStance2', 'LdoubleStance2Duration', 'Lduration', 'LsimpleStance', 'LsimpleStanceDuration', 
            # 'Lspeed', 'LstanceDuration', 'LstancePhase', 'LstepDuration', 'LstepLength', 'LstepPhase', 'LstrideLength', 'LstrideWidth', 'LswingDuration', 'LswingPhase', 
            # 'Rcadence', 'RdoubleStance1', 'RdoubleStance1Duration', 'RdoubleStance2', 'RdoubleStance2Duration', 'Rduration', 'RsimpleStance', 'RsimpleStanceDuration', 
            # 'Rspeed', 'RstanceDuration', 'RstancePhase', 'RstepDuration', 'RstepLength', 'RstepPhase', 'RstrideLength', 'RstrideWidth', 'RswingDuration', 'RswingPhase']


class NormativeStpHelper:
    def __init__(self):
        self.normativeDataPath = pyCGM2.NORMATIVE_DATABASE_PATH+"stp\\pyCGM2_GaitRite_normaldata.xlsx"
        self.m_df = pd.read_excel(self.normativeDataPath)

        self._matching_columns = {
            "speed": "Velocity (m.sec)",
            "cadence": "Cadence (steps.min)",
            "duration": "Stride Time (sec)",
            "stepDuration": "Step Time (sec)",
            "strideLength": "Stride Length (m)",
            "stepLength": "Step Length (m)",
            "strideWidth": "Stride Width (m)",
            "swingPhase": "Swing %",
            "stancePhase": "Stance %",
            "simpleStance": "Single Support %",
            # doubleStance1 and doubleStance2 are not directly available, but can be derived from stancePhase and simpleStance
        }   


    def getCollumns(self):
        return self.m_df.columns.tolist()

    def getMinMax(self,label,age):


            try:
                label = self._matching_columns[label]  # get the corresponding column label in the Excel file
            except KeyError:
                LOGGER.logger.error(f"Label '{label}' not found in the normative data columns.")
                return None

            else:
                pace="Normal"
            
                rowMale = self.m_df[(self.m_df["Gender"] == "Male") & (self.m_df["Age (min)"] <= age) & (self.m_df["Age (max)"] >= age) & (self.m_df["Pace"] == pace)]
                rowMale = rowMale.iloc[0] if not rowMale.empty else None

                rowFemale = self.m_df[(self.m_df["Gender"] == "Female") & (self.m_df["Age (min)"] <= age) & (self.m_df["Age (max)"] >= age) & (self.m_df["Pace"] == pace)]
                rowFemale = rowFemale.iloc[0] if not rowFemale.empty else None

                rangeMale   = [rowMale[f"{label}_min"],   rowMale[f"{label}_max"]]   if rowMale   is not None else None 
                rangeFemale = [rowFemale[f"{label}_min"], rowFemale[f"{label}_max"]] if rowFemale is not None else None


                ranges = [r for r in [rangeMale, rangeFemale] if r is not None]
                norm_range = [min(r[0] for r in ranges), max(r[1] for r in ranges)] if ranges else None

                return norm_range