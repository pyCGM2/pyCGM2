# -*- coding: utf-8 -*-
"""Skeleton of a pyCGM2 procedure (template for new features)."""
class AbstractFeatureProcedure(object):
    """abstract procedure """
    def __init__(self):
        pass
    def run(self):
        """Run the procedure (to be implemented by subclasses)."""
        pass


class ConcreteFeatureProcedure(AbstractFeatureProcedure):
    """Example of a concrete procedure summing two arguments."""
    def __init__(self):
        super(ConcreteFeatureProcedure, self).__init__()
    def run(self,arg1, arg2):
        """calculation
        """
        return arg1 + arg2


