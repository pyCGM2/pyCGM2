"""High-level function computing spatio-temporal parameters."""

from pyCGM2.Cycles import cycleBuilders, cycleFilter
from pyCGM2.Tools import btkTools

def computeSpatioTemporalParameters(datapath, filenames):
    """Compute the spatio-temporal parameters of each gait cycle of c3d files.

    Args:
        datapath (str): Folder path.
        filenames (List[str]): c3d filenames.

    Returns:
        dict: Parameters organised as ``out[filename][context][label]`` (list of values, one per cycle).
    """

    out = {}

    for  filename in filenames:
        acq = btkTools.smartReader(datapath + filename)   

        cycleBuilder = cycleBuilders.GaitCyclesBuilder([acq])
        
        cyclefilter = cycleFilter.CyclesFilter()
        cyclefilter.setBuilder(cycleBuilder)
        cycleCollection = cyclefilter.build()

        out[filename] = {}
        for cycleIt in cycleCollection.cycles:
                context = cycleIt.context

                if context not in out[filename]:
                    out[filename][context] = {}

                for label, value in cycleIt.stps.items():

                    if label not in out[filename][context]:
                        out[filename][context][label] = []

                    out[filename][context][label].append(value)
    return out