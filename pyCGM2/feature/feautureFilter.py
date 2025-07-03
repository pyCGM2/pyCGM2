# -*- coding: utf-8 -*-
class VeryBasicFilter(object):
    """ very basic skeleton of a filter 
    
    Examples:

    .. code-block:: python
        
        myFilter = VeryBasicFilter(1, 2)
        out = myFilter.run()
    """

    def __init__(self, arg1, arg2):
        self.m_arg1 = arg1
        self.m_arg2 = arg2


    def run(self):
        out = self.m_arg1 + self.m_arg1
        return out
    

class procedureFilter(object):
    """ Skeleton of a filter calling a procedure 

    Examples:

    .. code-block:: python
        
        myProcedure =  FeautureProcedure()
        myFilter = VeryBasicFilter(1, 2,myProcedure)
        out = myFilter.run()
    """

    def __init__(self, arg1, arg2, procedure):
        self.m_arg1 = arg1
        self.m_arg2 = arg2
        self.procedure = procedure


    def run(self):
        out = self.m_procedure.run(self.m_arg1,self.m_arg2)
        return out