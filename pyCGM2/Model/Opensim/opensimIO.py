# -*- coding: utf-8 -*-
from pyCGM2 import opensim4 as opensim
import numpy as np
import pandas as pd
pd.set_option("display.precision", 8)


class OpensimDataFrame(object):
    def __init__(self, DATA_PATH, filename, freq=100):

        self.m_DATA_PATH = DATA_PATH
        self.m_filename = filename

        storageObject = opensim.Storage(DATA_PATH+filename)
        osimlabels = storageObject.getColumnLabels()

        data = dict()

        self.m_header = ""
        with open(DATA_PATH+filename) as f:
            contents = f.readlines()
            for line in contents:
                if "endheader" in line:
                    break
                else:
                    self.m_header = self.m_header + line
        self.m_header = self.m_header + "endheader\n"

        for index in range(1, osimlabels.getSize()):  # 1 because 0 is time
            label = osimlabels.get(index)
            index_x = storageObject.getStateIndex(osimlabels.get(index))
            array_x = opensim.ArrayDouble()
            storageObject.getDataColumn(index_x, array_x)
            n = array_x.getSize()
            values = np.zeros((n))
            for i in range(0, n):
                values[i] = array_x.getitem(i)
            data[label] = values

        self.m_dataframe = pd.DataFrame(data)
        timevalues = np.arange(
            0, self.m_dataframe.shape[0]/freq, 1/freq)
        self.m_dataframe["time"] = timevalues

        first_column = self.m_dataframe.pop('time')
        self.m_dataframe.insert(0, 'time', first_column)

    def getDataFrame(self):
        return self.m_dataframe

    def save(self, outDir = None, filename=None):

        directory = self.m_DATA_PATH if outDir is None else  outDir
        filename = self.m_filename if filename is None else  filename

        file1 = open(directory + filename,"w")

        for it in self.m_header:
            file1.write(it)

        columns = self.m_dataframe.columns.to_list()
        for i in range(0, len(columns)):
            if i == len(columns)-1:
                file1.write(columns[i]+"\n")
            else:
                file1.write(columns[i]+"\t")

        for j in range(0, self.m_dataframe.shape[0]):
            li = self.m_dataframe.iloc[j].to_list()
            for k in range(0, len(li)):
                if k == len(li)-1:
                    file1.write("      %.8f\n" % (li[k])) if li[k] >= 0 else file1.write(
                        "     %.8f\n" % (li[k]))
                else:
                    file1.write("      %.8f\t" % (li[k])) if li[k] >= 0 else file1.write(
                        "     %.8f\t" % (li[k]))

                    # file1.write("      "+str(li[k])+"\n")
        file1.close()
