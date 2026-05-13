import pyCGM2
import moveck


def normalized_index(sequence, target):
    target = target.strip()
    for i, value in enumerate(sequence):
        if value.strip() == target:
            return i
    return None



class AbstractMekTrialTransformProcedure:
    def __init__(self):
        pass


class mekViconTrialTransformProcedure(AbstractMekTrialTransformProcedure):

    def __init__(self,cgmVersion):
        super(mekViconTrialTransformProcedure, self).__init__()
        self.m_cgmVersion = cgmVersion

    def run(self,datastorage,path, c3dFilename):
        root = datastorage.root()
        root.create_attribute('trialNamingMethod', 'stem')

        trial = moveck.import_trial(datastorage, path +c3dFilename)


        if root.exists_group(f"Trials/{c3dFilename[:-4]}/Devices"):
             root.delete_group(f"Trials/{c3dFilename[:-4]}/Devices")

        if root.exists_group(f"Trials/{c3dFilename[:-4]}/Processings"):
             root.delete_group(f"Trials/{c3dFilename[:-4]}/Processings")


        pointMetadatae = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Format/Metadata/POINT")
        labels = pointMetadatae.retrieve_attribute("LABELS").read()
        descriptions = pointMetadatae.retrieve_attribute("DESCRIPTIONS").read()

        split_spec = {
        "callable_unit": "data-modifier.set-split",
        "SourceSet": "Format/Data/Points",
        "DestinationGroup": "Devices/Mocap/Markers",
        "Filter": ["Types", "marker"]
        }
        moveck.transform_data(trial, split_spec)

        # spec = {
        #     "callable_unit": "c3d-format.detect-forceplate-channels",
        #     "SourceGroup": f"Trials/{c3dFilename[:-4]}",
        #     "DestinationGroup" : f"Trials/{c3dFilename[:-4]}/Devices/ForcePlateChannels",
        #     "extractionMethod": "split"
        # }
        # moveck.transform_data(trial, spec)


        # datastorage.dump("test.h5")

        grp = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/Markers")
        for label in  grp.list_set_children_name():
                grp.retrieve_set(label).create_attribute("MetricType","Position")

        if self.m_cgmVersion is not None:
            split_spec = {
                "callable_unit": "data-modifier.set-split",
                "SourceSet": "Format/Data/Points",
                "DestinationGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Angles",
                "Filter": ["Types", "angle"]
                }
            moveck.transform_data(trial, split_spec)
            grp = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Angles")
            for label in  grp.list_set_children_name():
                    grp.retrieve_set(label).create_attribute("MetricType","Angle")

            split_spec = {
                        "callable_unit": "data-modifier.set-split",
                        "SourceSet": "Format/Data/Points",
                        "DestinationGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Forces",
                        "Filter": ["Types", "force"]
                        }
            moveck.transform_data(trial, split_spec)
            grp = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Forces")
            for label in  grp.list_set_children_name():
                    grp.retrieve_set(label).create_attribute("MetricType","Force")

            split_spec = {
                        "callable_unit": "data-modifier.set-split",
                        "SourceSet": "Format/Data/Points",
                        "DestinationGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Moments",
                        "Filter": ["Types", "moment"]
                        }
            moveck.transform_data(trial, split_spec)
            grp = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Moments")
            for label in  grp.list_set_children_name():
                    grp.retrieve_set(label).create_attribute("MetricType","Moment")

            split_spec = {
                        "callable_unit": "data-modifier.set-split",
                        "SourceSet": "Format/Data/Points",
                        "DestinationGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Powers",
                        "Filter": ["Types", "power"]
                        }
            moveck.transform_data(trial, split_spec)
            grp = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Powers")
            for label in  grp.list_set_children_name():
                    grp.retrieve_set(label).create_attribute("MetricType","Power")


            split_spec = {
                        "callable_unit": "data-modifier.set-split",
                        "SourceSet": "Format/Data/Points",
                        "DestinationGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Modeled_markers",
                        "Filter": ["Types", "modeled_marker"]
                        }
            moveck.transform_data(trial, split_spec)
            grp = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Modeled_markers")
            for label in  grp.list_set_children_name():
                    grp.retrieve_set(label).create_attribute("MetricType","Position")


            split_spec = {
                        "callable_unit": "data-modifier.set-split",
                        "SourceSet": "Format/Data/Points",
                        "DestinationGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Usermodelouputs",
                        "Filter": ["Types", "usermo"]
                        }
            moveck.transform_data(trial, split_spec)


            if root.exists_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Usermodelouputs"):
                usermos = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Usermodelouputs")
                for label in  usermos.list_set_children_name():
                    if label in pyCGM2.MUSCLES_LABELS:
                        usermos.retrieve_set(label).create_attribute("MetricType","MuscleLength")
                        usermos.retrieve_set(label).create_attribute("Unit","m")


        # -------------------------copy-----------------------------------------------
        copy_marker_spec = {
            "SourceGroup": f"Devices/Mocap/Markers",
            "DestinationGroup": "Processings/Devices/Mocap/Markers",
            "callable_unit": "data-modifier.group-copy",
        }
        moveck.transform_data(trial, copy_marker_spec)
        

        if self.m_cgmVersion is not None:
            copy_marker_spec = {
                "SourceGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Angles",
                "DestinationGroup": f"Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Angles",
                "callable_unit": "data-modifier.group-copy",
            }
            moveck.transform_data(trial, copy_marker_spec)


            copy_marker_spec = {
                "SourceGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Forces",
                "DestinationGroup": f"Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Forces",
                "callable_unit": "data-modifier.group-copy",
            }
            moveck.transform_data(trial, copy_marker_spec)

            copy_marker_spec = {
                "SourceGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Moments",
                "DestinationGroup": f"Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Moments",
                "callable_unit": "data-modifier.group-copy",
            }
            moveck.transform_data(trial, copy_marker_spec)



            copy_marker_spec = {
                "SourceGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Powers",
                "DestinationGroup": f"Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Powers",
                "callable_unit": "data-modifier.group-copy",
            }
            moveck.transform_data(trial, copy_marker_spec)


            copy_marker_spec = {
                "SourceGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Modeled_markers",
                "DestinationGroup": f"Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Modeled_markers",
                "callable_unit": "data-modifier.group-copy",
            }
            moveck.transform_data(trial, copy_marker_spec)

            copy_marker_spec = {
                "SourceGroup": f"Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/Usermodelouputs",
                "DestinationGroup": f"Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/MusculoSkeletal/MuscleLength",
                "callable_unit": "data-modifier.group-copy",
                "Filter": ["MetricType", "MuscleLength"]
            }
            moveck.transform_data(trial, copy_marker_spec)

            # rename usermo sets with muscle names instead of generic labels
            if root.exists_group(f"Trials/{c3dFilename[:-4]}/Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/MusculoSkeletal/MuscleLength"):
                usermos = root.retrieve_group(f"Trials/{c3dFilename[:-4]}/Processings/Devices/Mocap/{self.m_cgmVersion}/ViconModelOutputs/MusculoSkeletal/MuscleLength")
                for label in  usermos.list_set_children_name():
                    
                    index = normalized_index(labels, label)
                    muscleLabel = descriptions[index].split(":")[0].removesuffix('[0]')
                    
                    values=  usermos.retrieve_set(label).read()
                    attrs={}
                    for it in usermos.retrieve_set(label).list_attributes_name():
                        attrs[it]= usermos.retrieve_set(label).retrieve_attribute(it).read()

                    usermos.delete_group(label)

                    usermos.create_set(muscleLabel,values)
                    for key in attrs: 
                        usermos.retrieve_set(muscleLabel).create_attribute(key,attrs[key])





class mekTrialTransformFilter:
    def __init__(self,datastorage,procedure=None):

        self.m_datastorage = datastorage
        self.m_procedure = procedure

    def run(self,path,c3dFilename):
        if self.m_procedure is not None:
            self.m_procedure.run(self.m_datastorage,path, c3dFilename)