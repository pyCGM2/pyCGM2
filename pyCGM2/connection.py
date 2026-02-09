class NexusConnection():
    def __init__(self): 

        self.NEXUS_PYTHON_CONNECTED = False    
        try:
            from viconnexusapi import ViconNexus 
            from pyCGM2.Nexus import nexusTools 

            self.NEXUS = ViconNexus.ViconNexus()
            self.nexusTools = nexusTools
            
        except:
            pass

    def isConnected(self):
        return self.NEXUS.Client.IsConnected()
    
