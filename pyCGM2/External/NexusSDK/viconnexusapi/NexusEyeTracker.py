class NexusEyeTracker:
    # Device details for an eye tracker in ViconNexus
    def __init__(self):
        self.SubjectName = str()  # Name of the Subject (can be blank)
        self.SegmentName = str()  # Name of the Segment (can be blank)
        self.EyePoseT = [0] * 3  # Eye pose translation in mm
        self.EyePoseR = [0] * 9  # Eye pose rotation matrix (row major format)
        self.Offset = [0] * 3  # Eye offset in mm
