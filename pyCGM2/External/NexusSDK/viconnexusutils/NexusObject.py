import abc


class NexusObject(object):
    """NexusObject An interface for objects which can be read from/written to Nexus.

     Base model output types should derive from this class and implement the
     abstract functions to correctly read/write data from/to Nexus"""

    __metaclass__ = abc.ABCMeta

    def __init__(self, subject_name=None):
        """class constructor The subject name is optional"""
        self._subject_name = subject_name

    @property
    def subject_name(self):
        return self._subject_name

    @subject_name.setter
    def subject_name(self, subject_name):
        """Sets the name of the subject to which this object's data belongs"""
        self._subject_name = subject_name

    @abc.abstractmethod
    def Create(self, name, SDK):
        """Creates a blank output"""

    @abc.abstractmethod
    def Read(self, name, SDK):
        """Reads an existing output from Nexus"""

    @abc.abstractmethod
    def Write(self, name, SDK):
        """Writes to an existing output in Nexus"""

    @staticmethod
    def _get_subject_name(a, b):

        if not isinstance(a, NexusObject) or not isinstance(b, NexusObject):
            raise TypeError('Both arguments must be NexusObjects')

        if a.subject_name == b.subject_name:
            return a.subject_name

        elif not b.subject_name:
            return a.subject_name

        elif not a.subject_name:
            return b.subject_name

        else:
            return ''
