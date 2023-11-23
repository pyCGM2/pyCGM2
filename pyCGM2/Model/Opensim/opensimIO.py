# -*- coding: utf-8 -*-
#from pyCGM2 import opensim4 as opensim

import pyCGM2; LOGGER = pyCGM2.LOGGER
try:
    import opensim
except:
    try:
        from pyCGM2 import opensim4 as opensim
    except:
        LOGGER.logger.error("[pyCGM2] opensim not found on your system")


import numpy as np
import pandas as pd
pd.set_option("display.precision", 8)

class ImuStorageFile(object):
    def __init__(self,DATA_PATH, filename, freq):

        self.m_DATA_PATH = DATA_PATH
        self.m_filename = filename
        self.m_freq = freq

        # Define the outputs
        self.m_header = pd.DataFrame([f'DataRate={freq}', 
                                    f'DataType=Quaternion',
                                    f'version=3', 
                                    f'OpenSimVersion=4.4', 
                                    f'endheader'])

        self.m_data = {}


    def setData(self, imuName, quaternionArray):
        self.m_data[imuName] = quaternionArray

         

    def construct(self,static=False):
        column_header = pd.DataFrame([f'time'] + [f"{name}_imu" for name in self.m_data]).T
        
        time = [0]
        if static:
            data_output = pd.DataFrame([f"{time[0]}"] + \
                                    [ f'{self.m_data[name][0,0]},  {self.m_data[name][0,1]}, {self.m_data[name][0,2]},  {self.m_data[name][0,3]}'
                                    for name in self.m_data]).T
        else: 
            time = np.array([np.divide([range(self.m_data[name].shape[0]) for name in self.m_data], self.m_freq)[0]]).T

            data = np.array([[f'{self.m_data[name][frame,0]}, {self.m_data[name][frame,1]}, '
                            f'{self.m_data[name][frame,2]}, {self.m_data[name][frame,3]}'
                            for frame in range(self.m_data[name].shape[0])] for name in self.m_data]).T

            data_output = pd.DataFrame(np.append(time, data, axis=1))

        with open(self.m_DATA_PATH+self.m_filename, 'w') as fp:
                fp.write(self.m_header.to_csv(index=False, header=False, lineterminator='\n'))
                fp.write(column_header.to_csv(index=False, sep='\t', header=False, lineterminator='\n'))
                fp.write(data_output.to_csv(index=False, sep='\t', header=False, lineterminator='\n'))


class OpensimDataFrame(object):
    def __init__(self, DATA_PATH, filename):

        self.m_DATA_PATH = DATA_PATH
        self.m_filename = filename

        storageObject = opensim.Storage(DATA_PATH+filename)
        lastTime = storageObject.getLastTime()

        osimlabels = storageObject.getColumnLabels()

        data = {}

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

        index_xTime = storageObject.getStateIndex("time")
        array_xTime = opensim.ArrayDouble()
        storageObject.getTimeColumn(array_xTime)
        freq = 1/(array_xTime.getitem(1)-array_xTime.getitem(0))

        timevalues = np.arange(
            0, self.m_dataframe.shape[0], 1)/freq
        
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
