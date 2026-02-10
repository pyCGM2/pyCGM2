
from pyCGM2.Cycles import cycleBuilders, cycleFilter
from pyCGM2.Tools import btkTools

def computeSpatioTemporalParameters(datapath, filenames):

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