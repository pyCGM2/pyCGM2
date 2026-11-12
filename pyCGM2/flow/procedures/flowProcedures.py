"""Base class of the *flow* procedures."""
class AbstractFlowProcedure(object):
    """abstract procedure """
    def __init__(self):
        pass
    def run(self):
        """Run the procedure (to be implemented by subclasses)."""
        pass