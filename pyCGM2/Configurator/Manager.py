# -*- coding: utf-8 -*-
import pyCGM2
from pyCGM2.Utils import files
from pyCGM2 import enums


# def keys_exists(element, *keys):
#     '''
#     Check if *keys (nested) exists in `element` (dict).
#     '''
#     if type(element) is not dict:
#         raise AttributeError('keys_exists() expects dict as first argument.')
#     if len(keys) == 0:
#         raise AttributeError('keys_exists() expects at least two arguments, one given.')
#
#     _element = element
#     for key in keys:
#         try:
#             _element = _element[key]
#         except KeyError:
#             return False
#     return True


class ConfigManager(object):
    def __init__(self,settings):
        self._userSettings = settings


class ModelConfigManager(ConfigManager):
    """

    """
    def __init__(self,settings):
        super(ModelConfigManager, self).__init__(settings)

        self._internSettings = None

        import ipdb; ipdb.set_trace()

    def getInternalSettings(self):
        return self._internSettings

    def getUserSettings(self):
        return self._userSettings

    def getUserSettings(self):
        return self._userSettings

    def update(self):
        pass


    @property
    def DATA_PATH(self):
        return  self._userSettings["DATA_PATH"]


    @property
    def staticTrial(self):
        return  self._userSettings["Calibration"]["StaticTrial"]

    @property
    def dynamicTrials(self):
        return  self._userSettings["Fitting"]["Trials"]



class CGM1ConfigManager(ModelConfigManager):
    """

    """
    def __init__(self,settings,internalSettingsFile = None, translatorFile=None):
        super(CGM1ConfigManager, self).__init__(settings)

        self._translatorFile = translatorFile
        self._internalSettingsFile = internalSettingsFile

        self.__internalsettings()

        # run data to overwrite in internalSettings
        self.leftFlatFoot
        self.rightFlatFoot
        self.markerDiameter
        self.pointSuffix
        self.translators

    def __internalsettings(self):
        if self._internalSettingsFile is None:
            self._internSettings = files.openConfigurationFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.ysettings")
        else:
            self._internSettings = files.openConfigurationFile(self._userSettings["DATA_PATH"],internalSettingsFile)




    @property
    def leftFlatFoot(self):
        value = self._userSettings["Calibration"]["Left flat foot"]
        self._internSettings["Calibration"]["Left flat foot"] = value # overwriting
        return value

    @property
    def rightFlatFoot(self):
        value = self._userSettings["Calibration"]["Right flat foot"]
        self._internSettings["Calibration"]["Right flat foot"] = value # overwriting
        return value

    @property
    def markerDiameter(self):
        value = self._userSettings["Global"]["Marker diameter"]
        self._internSettings["Global"]["Marker diameter"] = value # overwriting
        return  value

    @property
    def pointSuffix(self):
        value = self._userSettings["Global"]["Point Suffix"]
        self._internSettings["Global"]["Point Suffix"] = value # overwriting
        return  value

    @property
    def translators(self): # overwriting if Translators exist
        if self._translatorFile is not None:
            translators = files.openConfigurationFile(self._userSettings["DATA_PATH"],self._translatorFile)
            self._internSettings["Translators"] = translators
            return self._internSettings["Translators"]
        else:
            return self._internSettings["Translators"]

    @property
    def momentProjection(self):
        if self._internSettings["Fitting"]["Moment Projection"] == "Distal":
            return  enums.MomentProjection.Distal
        elif self._internSettings["Fitting"]["Moment Projection"] == "Proximal":
            return  enums.MomentProjection.Proximal
        elif self._internSettings["Fitting"]["Moment Projection"] == "Global":
            return  enums.MomentProjection.Global
        elif self._internSettings["Fitting"]["Moment Projection"] == "JCS":
            return enums.MomentProjection.JCS


class CGM1_1ConfigManager(CGM1ConfigManager):
    """

    """
    def __init__(self,settings,internalSettingsFile = None, translatorFile=None):
        super(CGM1_1ConfigManager, self).__init__(settings,internalSettingsFile = internalSettingsFile, translatorFile=translatorFile)

    def __internalsettings(self):
        if self._internalSettingsFile is None:
            self._internSettings = files.openConfigurationFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.ysettings")
        else:
            self._internSettings = files.openConfigurationFile(self._userSettings["DATA_PATH"],internalSettingsFile)

class CGM2_1ConfigManager(CGM1ConfigManager):
    """

    """
    def __init__(self,settings,internalSettingsFile = None, translatorFile=None):
        super(CGM1_1ConfigManager, self).__init__(settings,internalSettingsFile = internalSettingsFile, translatorFile=translatorFile)

        self.__internalsettings()

    def __internalsettings(self):
        if self._internalSettingsFile is None:
            self._internSettings = files.openConfigurationFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.ysettings")
        else:
            self._internSettings = files.openConfigurationFile(self._userSettings["DATA_PATH"],internalSettingsFile)


    @property
    def hjcMethod(self):
        return self._internSettings["Calibration"]["HJC"]

class CGM2_2ConfigManager(CGM1ConfigManager):
    """

    """
    def __init__(self,settings,internalSettingsFile = None, translatorFile=None, ikweightFile=None):
        super(CGM1_1ConfigManager, self).__init__(settings,internalSettingsFile = internalSettingsFile, translatorFile=translatorFile)

        self._ikweightFile = ikweightFile

        self.__internalsettings()

        # run data to overwrite in internalSettings
        self.ikWeight

    def __internalsettings(self):
        if self._internalSettingsFile is None:
            self._internSettings = files.openConfigurationFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_2-pyCGM2.ysettings")
        else:
            self._internSettings = files.openConfigurationFile(self._userSettings["DATA_PATH"],internalSettingsFile)



    @property
    def hjcMethod(self):
        return self._internSettings["Calibration"]["HJC"]


    @property
    def ikWeight(self): # overwriting if Translators exist
        if self._ikweightFile is not None:
            ikweight = files.openConfigurationFile(self._userSettings["DATA_PATH"],self._ikweightFile)
            self._internSettings["Fitting"]["Weight"] = ikweight
            return self._internSettings["Fitting"]["Weight"]
        else:
            return self._internSettings["Fitting"]["Weight"]

class CGM2_3ConfigManager(CGM1ConfigManager):
    """

    """
    def __init__(self,settings,internalSettingsFile = None, translatorFile=None, ikweightFile=None):
        super(CGM2_3ConfigManager, self).__init__(settings,internalSettingsFile = internalSettingsFile, translatorFile=translatorFile,ikweightFile=ikweightFile)

        self.__internalsettings()

    def __internalsettings(self):
        if self._internalSettingsFile is None:
            self._internSettings = files.openConfigurationFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.ysettings")
        else:
            self._internSettings = files.openConfigurationFile(self._userSettings["DATA_PATH"],internalSettingsFile)

    @property
    def enableIK(self):
        return self._internSettings["Global"]["EnableIK"]

class CGM2_4ConfigManager(CGM1ConfigManager):
    """

    """
    def __init__(self,settings,internalSettingsFile = None, translatorFile=None, ikweightFile=None):
        super(CGM2_4ConfigManager, self).__init__(settings,internalSettingsFile = internalSettingsFile, translatorFile=translatorFile,ikweightFile=ikweightFile)

        self.__internalsettings()

    def __internalsettings(self):
        if self._internalSettingsFile is None:
            self._internSettings = files.openConfigurationFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.ysettings")
        else:
            self._internSettings = files.openConfigurationFile(self._userSettings["DATA_PATH"],internalSettingsFile)
