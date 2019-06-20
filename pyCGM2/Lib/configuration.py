# -*- coding: utf-8 -*-
from pyCGM2.Configurator import EmgManager
from pyCGM2.Configurator import ModelManager

# -----emg------
def getEmgConfiguration(userSettings,internalSettings):

    # --- Manager ----
    manager = EmgManager.EmgConfigManager(userSettings,localInternalSettings=internalSettings)
    manager.contruct()

    return manager.getEmgConfiguration()


# -----Model------

def getCGMmanager(CGMversion,userSettings,internalSettings=None,translators=None,localIkWeight=None,vsk=None):

    # --- Manager ----
    if model  = "CGM1.0":
        manager = ModelManager.CGM1ConfigManager(userSettings,
                                                    localInternalSettings=internalSettings,
                                                    localTranslators=translators,
                                                    localIkWeight=localIkWeight,
                                                    vsk=vsk)

    elif model  = "CGM1.1":
        manager = ModelManager.CGM1_1ConfigManager(userSettings,
                                                    localInternalSettings=internalSettings,
                                                    localTranslators=translators,
                                                    localIkWeight=localIkWeight,
                                                    vsk=vsk)

    elif model  = "CGM2.1":
        manager = ModelManager.CGM2_1ConfigManager(userSettings,
                                                    localInternalSettings=internalSettings,
                                                    localTranslators=translators,
                                                    localIkWeight=localIkWeight,
                                                    vsk=vsk)

    elif model  = "CGM2.2":
        manager = ModelManager.CGM2_2ConfigManager(userSettings,
                                                    localInternalSettings=internalSettings,
                                                    localTranslators=translators,
                                                    localIkWeight=localIkWeight,

    elif model  = "CGM2.3":
        manager = ModelManager.CGM2_3ConfigManager(userSettings,
                                                    localInternalSettings=internalSettings,
                                                    localTranslators=translators,
                                                    localIkWeight=localIkWeight,
                                                    vsk=vsk)
                                                    vsk=vsk)
    elif model  = "CGM2.4":
        manager = ModelManager.CGM2_4ConfigManager(userSettings,
                                                    localInternalSettings=internalSettings,
                                                    localTranslators=translators,
                                                    localIkWeight=localIkWeight,
                                                    vsk=vsk)


    elif model  = "CGM2.5":
        manager = ModelManager.CGM2_5ConfigManager(userSettings,
                                                    localInternalSettings=internalSettings,
                                                    localTranslators=translators,
                                                    localIkWeight=localIkWeight,
                                                    vsk=vsk)
    else:
        raise Exception ("[pyCGM2] : Model version not known (choice CGM1.0 to CGM2.5 )")

    manager.contruct()

    return manager
