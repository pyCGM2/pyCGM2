# coding: utf-8
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spm1d

import pyCGM2
from pyCGM2.Processing import analysisHandler
from pyCGM2.Utils import utils
import pyCGM2.Math.normalisation  as MathNormalisation

def registrateAnalysisOnGaitPhases(analysisInstance):

    # TODO : cast as a filter !

    logging.warning("[pyCGM2] No registration of the descriptive stats values")

    stancePeriod=60.0
    swingPeriod=100.0-stancePeriod
    double1Period=10.0
    double2Period=10.0
    simplePeriod=stancePeriod-(double1Period+double2Period)

    section = analysisInstance.kinematicStats
    if section.data != {}:
        for key  in section.data.keys():
            label =  key[0]
            context =  key[1]

            cycleValues = section.data[label,context]["values"]
            registratedCycles = list()
            for i in range(len(cycleValues)):
                d1 = int(round(section.pst["doubleStance1",context]["values"][i]))
                d2 = int(round(section.pst["stancePhase",context]["values"][i] - analysisInstance.kinematicStats.pst["doubleStance2",context]["values"][i]))
                st = int(round(section.pst["stancePhase",context]["values"][i]))

                data1 = cycleValues[i][0:d1,:]
                data1r = MathNormalisation.timeSequenceNormalisation(10,data1)

                data2 = cycleValues[i][d1:d2,:]
                data2r = MathNormalisation.timeSequenceNormalisation(40,data2)

                data3 = cycleValues[i][d2:st,:]
                data3r = MathNormalisation.timeSequenceNormalisation(10,data3)

                data4 = cycleValues[i][st:101,:]
                data4r = MathNormalisation.timeSequenceNormalisation(41,data4)

                registratedCycles.append( np.concatenate((data1r,data2r,data3r,data4r)))
                logging.debug("[pyCGM2] : registration of cycle %i of the kinematic parameter [%s,%s]",(i,label,context))


            analysisInstance.kinematicStats.data[label,context]["values"] = registratedCycles

        section = analysisInstance.kineticStats
        if section.data != {}:
            for key  in section.data.keys():
                label =  key[0]
                context =  key[1]

                cycleValues = section.data[label,context]["values"]
                registratedCycles = list()
                for i in range(len(cycleValues)):
                    d1 = int(round(section.pst["doubleStance1",context]["values"][i]))
                    d2 = int(round(section.pst["stancePhase",context]["values"][i] - analysisInstance.kinematicStats.pst["doubleStance2",context]["values"][i]))
                    st = int(round(section.pst["stancePhase",context]["values"][i]))

                    data1 = cycleValues[i][0:d1,:]
                    data1r = MathNormalisation.timeSequenceNormalisation(10,data1)

                    data2 = cycleValues[i][d1:d2,:]
                    data2r = MathNormalisation.timeSequenceNormalisation(40,data2)

                    data3 = cycleValues[i][d2:st,:]
                    data3r = MathNormalisation.timeSequenceNormalisation(10,data3)

                    data4 = cycleValues[i][st:101,:]
                    data4r = MathNormalisation.timeSequenceNormalisation(41,data4)

                    registratedCycles.append( np.concatenate((data1r,data2r,data3r,data4r)))

                    logging.debug("[pyCGM2] : registration of cycle %i of the kinetic parameter [%s,%s]",(i,label,context))

                analysisInstance.kineticStats.data[label,context]["values"] = registratedCycles


        section = analysisInstance.emgStats
        if section.data != {}:
            for key  in section.data.keys():
                label =  key[0]
                context =  key[1]

                cycleValues = section.data[label,context]["values"]
                registratedCycles = list()
                for i in range(len(cycleValues)):
                    d1 = int(round(section.pst["doubleStance1",context]["values"][i]))
                    d2 = int(round(section.pst["stancePhase",context]["values"][i] - analysisInstance.kinematicStats.pst["doubleStance2",context]["values"][i]))
                    st = int(round(section.pst["stancePhase",context]["values"][i]))

                    data1 = cycleValues[i][0:d1,:]
                    data1r = MathNormalisation.timeSequenceNormalisation(10,data1)

                    data2 = cycleValues[i][d1:d2,:]
                    data2r = MathNormalisation.timeSequenceNormalisation(40,data2)

                    data3 = cycleValues[i][d2:st,:]
                    data3r = MathNormalisation.timeSequenceNormalisation(10,data3)

                    data4 = cycleValues[i][st:101,:]
                    data4r = MathNormalisation.timeSequenceNormalisation(41,data4)

                    registratedCycles.append( np.concatenate((data1r,data2r,data3r,data4r)))
                    logging.debug("[pyCGM2] : registration of cycle %i of the emg parameter [%s,%s]",(i,label,context))


                analysisInstance.emgStats.data[label,context]["values"] = registratedCycles





class SpmAnalysisFilter(object):
    """
    """

    def __init__(self,procedure):

        self.m_procedure = procedure

    def ConfigureData(self,analyses,conditionDescriptions):
        self.m_analysis = analyses
        self.m_condDescriptions = conditionDescriptions

        self.m_inputs = self.m_procedure.configureData(self.m_analysis)

    def conductTest(self,ntests=1):
        return self.m_procedure.test(self.m_inputs,self.m_condDescriptions,ntests)

    def plot(self,spmResultsFlag=False,**figOptions):
        self.m_procedure.plot(self.m_inputs,self.m_condDescriptions,figOptions)



# ------------------PROCEDURES -----------------------------------------------

class AbstractTwoVectorsSPM(object):
    """
    Abstract Builder
    """
    def __init__(self,label,axis,context):
        self.m_label = label
        self.m_context = context
        self.m_axis = "XYZ".find(axis) if axis is not None else 0


    def configureData(self,analyses):

        label = self.m_label
        context = self.m_context

        inputs = list()

        # check if all labels/context exist in analysis
        for analysisIt in analyses:
            analysisHandler.isKeyExist(analysisIt,label,context, exceptionMode=True)

        Y=list()
        for analysisIt in analyses:
            n_cycle = analysisHandler.getNumberOfCycle(analysisIt,label,context)
            values = analysisHandler.getValues(analysisIt,label,context)
            for i  in range(0,n_cycle):
                Y.append(values[i][:,self.m_axis])

            inputs.append(np.asarray(Y))

        return inputs

    def test(self):
        pass

    def plot(self):
        pass




class SPM_ttest2Procedure(AbstractTwoVectorsSPM):
    """

    """

    def __init__(self,label,axis,context,**options):

        super(SPM_ttest2Procedure, self).__init__(label,axis,context)

        self.m_equal_var = True
        self.m_two_tailed = True
        self.m_interp = True

        if options.has_key("equal_var"): self.m_equal_var = options["equal_var"]
        if options.has_key("two_tailed"): self.m_two_tailed = options["two_tailed"]
        if options.has_key("interp"): self.m_interp = options["interp"]


    def test(self,inputs,conditionDescriptions,ntests):

        alpha = 0.05 if ntests==1 else spm1d.util.p_critical_bonf(0.05, ntests)
        self.m_test = spm1d.stats.ttest2(inputs[0], inputs[1], equal_var=self.m_equal_var)
        self.m_testi = self.m_test.inference(alpha, two_tailed=self.m_two_tailed, interp=self.m_interp)

        return self.m_testi.h0reject, self.m_testi.clusters



    def plot(self,inputs,conditionDescription,figOptions ):

        plt.close('all')
        pass


class SPM_t2HotellingProcedure(object):
    """

    """

    def __init__(self,vectors,**options):

        self.m_vectors = vectors

        self.m_equal_var = True
        self.m_two_tailed = True
        self.m_interp = True

        if options.has_key("equal_var"): self.m_equal_var = options["equal_var"]
        if options.has_key("two_tailed"): self.m_two_tailed = options["two_tailed"]
        if options.has_key("interp"): self.m_interp = options["interp"]

    def configureData(self,analyses):

        # check presence of data  and initiate Y vectors
        n_responses = len(self.m_vectors)
        n_nodes = 101


        # check if all vectors exist in analysis
        for analysisIt in analyses:
            for vectorIt in self.m_vectors:
                label = vectorIt[0]
                context = vectorIt[2]
                analysisHandler.isKeyExist(analysisIt,label,context, exceptionMode=True)


        # initate Data
        inputs = list()
        analysisCount = 0
        for analysisIt in analyses:
            vectCount = 0
            n_cycles = list()
            for vectorIt in self.m_vectors:
                label = vectorIt[0]
                context = vectorIt[2]
                n_cycle = analysisHandler.getNumberOfCycle(analysisIt,label,context)
                n_cycles.append(n_cycle)
            if utils.checkSimilarElement(n_cycles):
                n_cycle = utils.getSimilarElement(n_cycles)
                inputs.append(np.zeros((n_cycle,n_nodes,n_responses)))
            analysisCount+=1

        # TODO : check if n responses are equal

        # Popluate data
        analysisCount = 0
        for analysisIt in analyses:
            vectCount =0
            for vectorIt in self.m_vectors:
                label = vectorIt[0]
                axis = "XYZ".find(vectorIt[1]) if vectorIt[1] is not None else 0
                context = vectorIt[2]
                n_cycle = analysisHandler.getNumberOfCycle(analysisIt,label,context)
                values = analysisHandler.getValues(analysisIt,label,context)
                for i  in range(0,n_cycle):
                    inputs[analysisCount][i,:,vectCount]=values[i][:,axis]
                vectCount+=1

            analysisCount+=1

        return inputs


    def test(self,inputs,conditionDescriptions,ntests):

        alpha = 0.05 if ntests==1 else spm1d.util.p_critical_bonf(0.05, ntests)
        self.m_test = spm1d.stats.hotellings2(inputs[0], inputs[1], equal_var=self.m_equal_var)
        self.m_testi = self.m_test.inference(alpha)

        clusters = list()
        for cluster in self.m_testi.clusters:
            clusters.append([cluster.endpoints[0],cluster.endpoints[1]])


        return self.m_testi.h0reject, clusters



    def plot(self,inputs,conditionDescription,figOptions ):

        pass
