def exportJson_Schwartz2008(path,name="Schwartz2008.json"):
    """generate json from schwartz2008 dataset

    :param path: path
    :type path: string
    :param name: filename
    :type name: string
    :return: json file
    :rtype: file

    normativeDatasets.exportJson_Schwartz2008(DATA_PATH,name="test.json")

    """


    def constructJson(dataset, dataset_label, modalityLabel,data):

        dataset[dataset_label].update({modalityLabel:dict()})

        for label in data.keys():

            mean = data[label]["mean"]
            sd = data[label]["sd"]

            label = label.replace(".","")
            dataset[dataset_label][modalityLabel].update({label: {'X':None,'Y':None,'Z':None }})

            outX = list()
            for i in range(0,mean.shape[0]):
                values = [i*2, mean[i,0]-sd[i,0],mean[i,0]+sd[i,0]]
                outX.append(values)

            outY = list()
            for i in range(0,mean.shape[0]):
                values = [i*2, mean[i,1]-sd[i,1],mean[i,1]+sd[i,1]]
                outY.append(values)

            outZ = list()
            for i in range(0,mean.shape[0]):
                values = [i*2, mean[i,2]-sd[i,2],mean[i,2]+sd[i,2]]
                outZ.append(values)

            dataset[dataset_label][modalityLabel][label]["X"] = outX
            dataset[dataset_label][modalityLabel][label]["Y"] = outY
            dataset[dataset_label][modalityLabel][label]["Z"] = outZ

    dataset = {"Schwartz2008": dict()}

    nds = Schwartz2008("VerySlow")
    nds.constructNormativeData()
    constructJson(dataset,"Schwartz2008","VerySlow",nds.data)

    nds = Schwartz2008("Slow")
    nds.constructNormativeData()
    constructJson(dataset,"Schwartz2008","Slow",nds.data)

    nds = Schwartz2008("Free")
    nds.constructNormativeData()
    constructJson(dataset,"Schwartz2008","Free",nds.data)

    nds = Schwartz2008("Fast")
    nds.constructNormativeData()
    constructJson(dataset,"Schwartz2008","Fast",nds.data)

    nds = Schwartz2008("VeryFast")
    nds.constructNormativeData()
    constructJson(dataset,"Schwartz2008","VeryFast",nds.data)

    files.saveJson(path, name, dataset)







class Pinzone2014(object):

    def __init__(self,centre):
        """
        **Description :** Constructor of Pinzone2014_normativeDataBases


        :Parameters:
             - `centre` (str) - two choices : CentreOne or CentreTwo

        **Usage**

        .. code:: python

            from pyCGM2.Report import normativeDatabaseProcedure
            nd = normativeDatabaseProcedure.Pinzone2014_normativeDataBases("CentreOne")
            nd.constructNormativeData() # this function
            nd.data # dictionary with all parameters extracted from the dataset CentreOne reference in Pinzone2014


        """

        self.m_filename = pyCGM2.NORMATIVE_DATABASE_PATH+"Pinzone 2014\\Formatted- Pinzone2014.xlsx"
        self.m_centre = centre
        self.data = dict()

    def __setDict(self,dataframe,JointLabel,axisLabel, dataType):
        """ populate an item of the member dictionary (data)

        """

        if self.m_centre == "CentreOne":
            meanLabel = "CentreOneAverage"
            sdLabel = "CentreOneSD"

        elif self.m_centre == "CentreTwo":
            meanLabel = "CentreTwoAverage"
            sdLabel = "CentreTwoSD"
        else:
            raise Exception("[pyCGM2] - dont find Pinzone Normative data centre")

        if dataType == "Angles":
            self.data[JointLabel]= dict()

            data_X=dataframe[(dataframe.Angle == axisLabel[0])][meanLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][meanLabel].values if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][meanLabel].values if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T

            data_X=dataframe[(dataframe.Angle == axisLabel[0])][sdLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][sdLabel].values  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][sdLabel].values  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T

        if dataType == "Moments":
            self.data[JointLabel]= dict()

            data_X=dataframe[(dataframe.Moment == axisLabel[0])][meanLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][meanLabel].values if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][meanLabel].values if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T*1000.0

            data_X=dataframe[(dataframe.Moment == axisLabel[0])][sdLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][sdLabel].values  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][sdLabel].values  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T*1000.0

        if dataType == "Powers":
            self.data[JointLabel]= dict()

            data_X=dataframe[(dataframe.Power == axisLabel[0])][meanLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][meanLabel].values if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][meanLabel].values if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T

            data_X=dataframe[(dataframe.Power == axisLabel[0])][sdLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][sdLabel].values  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][sdLabel].values  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T


    def constructNormativeData(self):

        """
            **Description :**  Read initial xls file and construct the member dictionary (data)
        """

        angles =pd.read_excel(self.m_filename,sheet_name = "Angles")
        moments =pd.read_excel(self.m_filename,sheet_name = "Moments")
        powers =pd.read_excel(self.m_filename,sheet_name = "Powers")

        self.__setDict(angles,"Pelvis.Angles",["Pelvis Ant/Pst", "Pelvic Up/Dn", "Pelvic Int/Ext" ], "Angles")
        self.__setDict(angles,"Hip.Angles",["Hip Flx/Ext", "Hip Add/Abd", "Hip Int/Ext" ], "Angles")
        self.__setDict(angles,"Knee.Angles",["Knee Flx/Ext",None, None], "Angles")
        self.__setDict(angles,"Ankle.Angles",["Ankle Dor/Pla", None, None], "Angles")
        self.__setDict(angles,"Foot.Angles",[None, None, "Foot Int/Ext"], "Angles")

        self.__setDict(moments,"Hip.Moment",["Hip Extensor Moment ", "Hip Abductor Moment ", None ], "Moments")
        self.__setDict(moments,"Knee.Moment",["Knee Extensor Moment ", "Knee Abductor Moment ", None ], "Moments")
        self.__setDict(moments,"Ankle.Moment",["Plantarflexor Moment ", None, "Ankle Rotation Moment"], "Moments")

        self.__setDict(powers,"Hip.Power",[ None, None,"Hip Power" ], "Powers")
        self.__setDict(powers,"Knee.Power",[None, None,"Knee Power"], "Powers")
        self.__setDict(powers,"Ankle.Power",[None, None,"Ankle Power"], "Powers")


class Schwartz2008(object):


    def __init__(self,speed):

        """
        **Description :** Constructor of Schwartz2008_normativeDataBases

        :Parameters:
               - `speed` (str) -  choices : VerySlow, Slow, Free, Fast, VeryFast

        **usage**

        .. code:: python

            from pyCGM2.Report import normativeDatabaseProcedure
            nd = normativeDatabaseProcedure.Schwartz2008_normativeDataBases("Free")
            nd.constructNormativeData() # this function
            nd.data # dictionary with all parameters extracted from the dataset CentreOne reference in Pinzone2014


        """

        self.m_filename = pyCGM2.NORMATIVE_DATABASE_PATH+"Schwartz 2008\\Formatted- Schwartz2008.xlsx"

        self.m_speedModality = speed
        self.data = dict()

    def __setDict(self,dataframe,JointLabel,axisLabel, dataType):
        """
            Populate an item of the member dictionary (data)

        """


        if self.m_speedModality == "VerySlow":
            meanLabel = "VerySlowMean"
            sdLabel = "VerySlowSd"
        elif self.m_speedModality == "Slow":
            meanLabel = "SlowMean"
            sdLabel = "SlowSd"
        elif self.m_speedModality == "Free":
            meanLabel = "FreeMean"
            sdLabel = "FreeSd"
        elif self.m_speedModality == "Fast":
            meanLabel = "FastMean"
            sdLabel = "FastSd"
        elif self.m_speedModality == "VeryFast":
            meanLabel = "VeryFastMean"
            sdLabel = "VeryFastSd"



        if dataType == "Angles":
            self.data[JointLabel]= dict()
            data_X=dataframe[(dataframe.Angle == axisLabel[0])][meanLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][meanLabel].values if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][meanLabel].values if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T

            data_X=dataframe[(dataframe.Angle == axisLabel[0])][sdLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Angle == axisLabel[1])][sdLabel].values  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Angle == axisLabel[2])][sdLabel].values  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T

        if dataType == "Moments":
            self.data[JointLabel]= dict()

            data_X=dataframe[(dataframe.Moment == axisLabel[0])][meanLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][meanLabel].values if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][meanLabel].values if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T*1000.0

            data_X=dataframe[(dataframe.Moment == axisLabel[0])][sdLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Moment == axisLabel[1])][sdLabel].values  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Moment == axisLabel[2])][sdLabel].values  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T*1000.0

        if dataType == "Powers":
            self.data[JointLabel]= dict()

            data_X=dataframe[(dataframe.Power == axisLabel[0])][meanLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][meanLabel].values if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][meanLabel].values if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["mean"]= np.array([data_X,data_Y,data_Z]).T

            data_X=dataframe[(dataframe.Power == axisLabel[0])][sdLabel].values if axisLabel[0] is not None else np.zeros((51))
            data_Y=dataframe[(dataframe.Power == axisLabel[1])][sdLabel].values  if axisLabel[1] is not None else np.zeros((51))
            data_Z=dataframe[(dataframe.Power == axisLabel[2])][sdLabel].values  if axisLabel[2] is not None else np.zeros((51))
            self.data[JointLabel]["sd"] = np.array([data_X,data_Y,data_Z]).T


    def constructNormativeData(self):
        """
            **Description :**  Read initial xls file and construct the member dictionary (data)
        """

        angles =pd.read_excel(self.m_filename,sheet_name = "Joint Rotations")
        moments =pd.read_excel(self.m_filename,sheet_name = "Joint Moments")
        powers =pd.read_excel(self.m_filename,sheet_name = "Joint Power")


        self.__setDict(angles,"Pelvis.Angles",["Pelvic Ant/Posterior Tilt", "Pelvic Up/Down Obliquity", "Pelvic Int/External Rotation" ], "Angles")
        self.__setDict(angles,"Hip.Angles",["Hip Flex/Extension", "Hip Ad/Abduction", "Hip Int/External Rotation" ],"Angles")
        self.__setDict(angles,"Knee.Angles",["Knee Flex/Extension","Knee Ad/Abduction", "Knee Int/External Rotation"], "Angles")
        self.__setDict(angles,"Ankle.Angles",["Ankle Dorsi/Plantarflexion", None, None], "Angles")
        self.__setDict(angles,"Foot.Angles",[None, None, "Foot Int/External Progression"], "Angles")

        self.__setDict(moments,"Hip.Moment",["Hip Ext/Flexion", "Hip Ab/Adduction", None ], "Moments")
        self.__setDict(moments,"Knee.Moment",["Knee Ext/Flexion", "Knee Ab/Adduction", None ], "Moments")
        self.__setDict(moments,"Ankle.Moment",["Ankle Dorsi/Plantarflexion", None, None], "Moments")

        self.__setDict(powers,"Hip.Power",[ None, None,"Hip" ], "Powers")
        self.__setDict(powers,"Knee.Power",[None, None,"Knee"], "Powers")
        self.__setDict(powers,"Ankle.Power",[None, None,"Ankle"], "Powers")
