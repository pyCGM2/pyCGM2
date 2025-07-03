# -*- coding: utf-8 -*-
class AbstractFeatureProcedure(object):
    """abstract procedure """
    def __init__(self):
        pass
    def run(self):
        pass


class ConcreteFeatureProcedure(AbstractFeatureProcedure):
    """    """
    def __init__(self):
        super(ConcreteFeatureProcedure, self).__init__()
    def run(self,arg1, arg2):
        """calculation
        """
        return arg1 + arg2


