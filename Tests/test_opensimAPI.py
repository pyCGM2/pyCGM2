import opensim
import numpy as np

import pyCGM2
from pyCGM2.Tools import opensimTools

class Test_osim:
    def test_osimInterface(self): # Charger le modèle

        model = opensim.Model(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\"+"pycgm2-gait2392_simbody.osim")
        l1 = opensimTools.calculateSegmentLength(model,"hip_r","knee_r")

        print(f"Longueur du fémur droit : {l1} m")


        # model = osim.Model(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "interface\\CGM23\\"+"pycgm2-gait2392_simbody.osim")
        # state= model.initSystem()

        # # # Accéder aux points du fémur droit
        # # state = model.updWorkingState()
        # # femur_r = model.getBodySet().get("femur_r")

        # # Identifier les repères (par exemple, hanche et genou)
        # hip_joint = model.getJointSet().get("hip_r")
        # knee_joint = model.getJointSet().get("knee_r")
        # ankle_joint = model.getJointSet().get("ankle_r")

        # zero_vec = osim.Vec3(0, 0, 0)

        # # Obtenir les coordonnées des articulations
        # hip_location = hip_joint.getParentFrame().findStationLocationInGround(state, zero_vec)
        # knee_location = knee_joint.getParentFrame().findStationLocationInGround(state, zero_vec)
        # ankle_location = ankle_joint.getParentFrame().findStationLocationInGround(state, zero_vec)

        # # Convertir les SimTK::Vec3 en numpy arrays
        # hip_location_np = np.array([hip_location.get(i) for i in range(3)])
        # knee_location_np = np.array([knee_location.get(i) for i in range(3)])
        # ankle_location_np = np.array([ankle_location.get(i) for i in range(3)])

        # # Calculer la distance entre les positions
        # femur_length = np.linalg.norm(knee_location_np - hip_location_np)
        # tibia_length = np.linalg.norm(knee_location_np - ankle_location_np)

        # print(f"Longueur du fémur droit : {femur_length} m")
        # print(f"Longueur du tibia droit : {tibia_length} m")