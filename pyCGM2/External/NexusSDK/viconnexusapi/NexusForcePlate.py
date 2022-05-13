class NexusForcePlate:
    # Device details for a force plate in ViconNexus
    def __init__(self):
        self.LocalR = [0] * 9  # Local rotation matrix (row major format)
        self.LocalT = [0] * 3  # Local translation from the device origin in mm
        self.WorldR = [0] * 9  # World rotation matrix (row major format)
        self.WorldT = [0] * 3  # World translation in mm
        self.LowerBounds = [0] * 3  # lower bounds of the plate
        self.UpperBounds = [0] * 3  # upper bounds of the plate
        self.Context = ''
