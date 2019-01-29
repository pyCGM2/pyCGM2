# -*- coding: utf-8 -*-
class ConfigManager(object):
    def __init__(self,settings):
        self._userSettings = settings
        self._internSettings = None
        self.finalSettings = None

    def getFinalSettings(self):
        return self.finalSettings

    def getInternalSettings(self):
        return self._internSettings

    def getUserSettings(self):
        return self._userSettings
