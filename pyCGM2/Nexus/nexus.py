"""Connection to Vicon Nexus."""
class NexusConnection():
    """Connection to a running Vicon Nexus instance.

    If the connection succeeds, ``NEXUS`` (ViconNexus instance) and ``nexusTools`` are available
    and ``NEXUS_PYTHON_CONNECTED`` is set to True.
    """
    def __init__(self): 

        self.NEXUS_PYTHON_CONNECTED = False    
        try:
            from viconnexusapi import ViconNexus 
            from pyCGM2.Nexus import nexusTools 

            self.NEXUS = ViconNexus.ViconNexus()
            self.nexusTools = nexusTools
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass

    def isConnected(self):
        """Check the connection to Nexus.

        Returns:
            bool: True if the client is connected.
        """
        return self.NEXUS.Client.IsConnected()
    