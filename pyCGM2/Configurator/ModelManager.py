# -*- coding: utf-8 -*-
import pyCGM2
from pyCGM2.Configurator import Manager
from pyCGM2.Utils import files
from pyCGM2 import enums
import logging
import copy
from pyCGM2.Eclipse import vskTools,eclipse


class ModelConfigManager(Manager.ConfigManager):
    """

    """
    def __init__(self,settings):
        super(ModelConfigManager, self).__init__(settings)


    @property
    def staticTrial(self):
        return  self._userSettings["Calibration"]["StaticTrial"]

    @property
    def dynamicTrials(self):
        return  self._userSettings["Fitting"]["Trials"]

    @property
    def listOfdynamicTrials(self):
        li =list()
        for it in self._userSettings["Fitting"]["Trials"]:
            li.append(it["File"])

        return li

    def contruct(self):

        finalSettings =  copy.deepcopy(self._internSettings)

        finalSettings["Calibration"].update({"StaticTrial":self._userSettings["Calibration"]["StaticTrial"]})
        finalSettings["Fitting"].update({"Trials":self._userSettings["Fitting"]["Trials"]})

        for key in self._userSettings.keys(): #upate of #mp
            if key  not in finalSettings.keys():
                print key
                finalSettings.update({key : self._userSettings[key]})

        self.finalSettings = finalSettings




class CGM1ConfigManager(ModelConfigManager):
    """

    """
    def __init__(self,settings,localInternalSettings = None, localTranslators=None,vsk=None):
        super(CGM1ConfigManager, self).__init__(settings)

        self._localTranslators = localTranslators
        self._localInternalSettings = localInternalSettings

        self.__internalsettings()

        # run data to overwrite in internalSettings
        self.leftFlatFoot
        self.rightFlatFoot
        self.headFlat
        self.markerDiameter
        self.pointSuffix
        self.translators
        self.momentProjection

        if vsk is not None:
            self._vsk = vsk
        else:
            self._vsk = None

        self.requiredMp
        self.optionalMp


    def __internalsettings(self):
        if self._localInternalSettings is None:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM1-pyCGM2.settings"):
                self._internSettings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")
            else:
                self._internSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM1-pyCGM2.settings")
        else:
            logging.info("Local internal setting found")
            self._internSettings = self._localInternalSettings


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
    def headFlat(self):
        value = self._userSettings["Calibration"]["Head flat"]
        self._internSettings["Calibration"]["Head flat"] = value # overwriting
        return value

    @property
    def markerDiameter(self):
        value = self._userSettings["Global"]["Marker diameter"]
        self._internSettings["Global"]["Marker diameter"] = value # overwriting
        return  value

    @property
    def pointSuffix(self):
        value = self._userSettings["Global"]["Point suffix"]
        self._internSettings["Global"]["Point suffix"] = value # overwriting

        return  None if value=="None" else value

    @property
    def translators(self): # overwriting if Translators exist
        if self._localTranslators is not None:
            translators = self._localTranslators
            self._internSettings["Translators"] = translators["Translators"]
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

    @property
    def requiredMp(self):
        if self._vsk is None:
            return  self._userSettings["MP"]["Required"]
        else:
            required_mp,optional_mp = vskTools.getFromVskSubjectMp(self._vsk, resetFlag=True)
            self._userSettings["MP"]["Required"].update(required_mp)
            return required_mp

    @property
    def optionalMp(self):
        if self._vsk is None:
            return  self._userSettings["MP"]["Optional"]
        else:
            required_mp,optional_mp = vskTools.getFromVskSubjectMp(self._vsk, resetFlag=True)
            self._userSettings["MP"]["Optional"].update(optional_mp)
            return optional_mp



    def updateMp(self,model):

        if self.finalSettings is not None:
            self.finalSettings["MP"]["Required"].update(model.mp)
            self.finalSettings["MP"]["Optional"].update(model.mp_computed)



class CGM1_1ConfigManager(CGM1ConfigManager):
    """

    """
    def __init__(self,settings,localInternalSettings = None, localTranslators=None,vsk=None):
        super(CGM1_1ConfigManager, self).__init__(settings,localInternalSettings = localInternalSettings, localTranslators=localTranslators, vsk=vsk)

    def __internalsettings(self):
        if self._localInternalSettings is None:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM1_1-pyCGM2.settings"):
                self._internSettings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")
            else:
                self._internSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM1_1-pyCGM2.settings")

        else:
            self._internSettings = self._localInternalSettings

class CGM2_1ConfigManager(CGM1ConfigManager):
    """

    """
    def __init__(self,settings,localInternalSettings = None, localTranslators=None,vsk=None):
        super(CGM2_1ConfigManager, self).__init__(settings,localInternalSettings = localInternalSettings, localTranslators=localTranslators,vsk=vsk)

        self.__internalsettings()

    def __internalsettings(self):
        if self._localInternalSettings is None:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_1-pyCGM2.settings"):
                self._internSettings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")
            else:
                self._internSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_1-pyCGM2.settings")
        else:
            self._internSettings = self._localInternalSettings


    @property
    def hjcMethod(self):
        return self._internSettings["Calibration"]["HJC"]

class CGM2_2ConfigManager(CGM2_1ConfigManager):
    """

    """
    def __init__(self,settings,localInternalSettings = None, localTranslators=None, localIkWeight=None,vsk=None):
        super(CGM2_2ConfigManager, self).__init__(settings,localInternalSettings = localInternalSettings, localTranslators=localTranslators,vsk=vsk)

        self._localIkweight = localIkWeight

        self.__internalsettings()

        # run data to overwrite in internalSettings
        self.ikWeight

    def __internalsettings(self):
        if self._localInternalSettings is None:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_2-pyCGM2.settings"):
                self._internSettings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_2-pyCGM2.settings")
            else:
                self._internSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_2-pyCGM2.settings")
        else:
            self._internSettings = self._localInternalSettings



    @property
    def hjcMethod(self):
        return self._internSettings["Calibration"]["HJC"]


    @property
    def ikWeight(self): # overwriting if Translators exist
        if self._localIkweight is not None:
            self._internSettings["Fitting"]["Weight"] = self._localIkweight
            return self._internSettings["Fitting"]["Weight"]
        else:
            return self._internSettings["Fitting"]["Weight"]

class CGM2_3ConfigManager(CGM2_2ConfigManager):
    """

    """
    def __init__(self,settings,localInternalSettings = None, localTranslators=None, localIkWeight=None,vsk=None):
        super(CGM2_3ConfigManager, self).__init__(settings,localInternalSettings = localInternalSettings, localTranslators=localTranslators,
                                                localIkWeight=localIkWeight,vsk=vsk)

        self.__internalsettings()

    def __internalsettings(self):
        if self._localInternalSettings is None:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_3-pyCGM2.settings"):
                self._internSettings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.settings")
            else:
                self._internSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_3-pyCGM2.settings")
        else:
            self._internSettings = self._localInternalSettings

    @property
    def enableIK(self):
        return self._internSettings["Global"]["EnableIK"]

class CGM2_4ConfigManager(CGM2_3ConfigManager):
    """

    """
    def __init__(self,settings,localInternalSettings = None, localTranslators=None, localIkWeight=None,vsk=None):
        super(CGM2_4ConfigManager, self).__init__(settings,localInternalSettings = localInternalSettings, localTranslators=localTranslators,localIkWeight=localIkWeight,vsk=vsk)

        self.__internalsettings()

    def __internalsettings(self):
        if self._localInternalSettings is None:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_4-pyCGM2.settings"):
                self._internSettings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")
            else:
                self._internSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_4-pyCGM2.settings")

        else:
            self._internSettings = self._localInternalSettings


class CGM2_5ConfigManager(CGM2_4ConfigManager):
    """

    """
    def __init__(self,settings,localInternalSettings = None, localTranslators=None, localIkWeight=None,vsk=None):
        super(CGM2_5ConfigManager, self).__init__(settings,localInternalSettings = localInternalSettings, localTranslators=localTranslators,localIkWeight=localIkWeight,vsk=vsk)

        self.__internalsettings()

    def __internalsettings(self):
        if self._localInternalSettings is None:
            if os.path.isfile(pyCGM2.PYCGM2_APPDATA_PATH + "CGM2_5-pyCGM2.settings"):
                self._internSettings = files.openFile(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_5-pyCGM2.settings")
            else:
                self._internSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"CGM2_5-pyCGM2.settings")
        else:
            self._internSettings = self._localInternalSettings
