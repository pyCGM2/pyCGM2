import copy

def normalizedMuscleLength_withPose(analysisInstance, referenceDataframe):

    for keys in list(analysisInstance.muscleGeometryStats.data):
        if "MuscleLength" in keys[0]:
            muscle =  keys[0][0:keys[0].find("[")]
            value0 = referenceDataframe[muscle][0]
            newLabel = keys[0]+"_PoseNormalized" 
            
            toCopy = analysisInstance.muscleGeometryStats.data[keys[0],keys[1]]
            ncycle = len(toCopy["values"])

            analysisInstance.muscleGeometryStats.data[newLabel,keys[1]] = toCopy
    

            for cycle in range(0,ncycle):
                cycleValue = toCopy["values"][cycle][:,0]
                analysisInstance.muscleGeometryStats.data[newLabel,keys[1]]["values"][cycle][:,0] = cycleValue/value0
            
            meanValue = toCopy["mean"][:,0]
            meanStd = toCopy["std"][:,0]
            medianValue = toCopy["median"][:,0]

            analysisInstance.muscleGeometryStats.data[newLabel,keys[1]]["mean"][:,0] = meanValue/value0
            analysisInstance.muscleGeometryStats.data[newLabel,keys[1]]["std"][:,0] = meanStd/value0
            analysisInstance.muscleGeometryStats.data[newLabel,keys[1]]["median"][:,0] = medianValue/value0

        # analysisInstance.muscleGeometryInfo["NormalizedLength"]=True