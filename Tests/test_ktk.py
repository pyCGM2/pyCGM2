# coding: utf-8

# pytest -s --disable-pytest-warnings test_ktk.py::Test_ktk::test_TimeseriesFelixDemo

import pandas as pd
from pyCGM2.Utils import files

from pyCGM2.External.ktk.kineticstoolkit import timeseries
from pyCGM2.External.ktk.kineticstoolkit import cycles
from pyCGM2.External.ktk.kineticstoolkit import files
import  pyCGM2.External.ktk.kineticstoolkit  as ktk
import numpy as np

import matplotlib.pyplot as plt



class Test_ktk:
    def test_TimeseriesFelixDemo(self):

        # --- basic --- 
        ts = timeseries.TimeSeries()
        ts.time = np.arange(0, 10, 0.1)  # 10 seconds at 10 Hz
        ts.data["Sinus"] = np.sin(ts.time)
        ts = ts.add_event(8.56, "push")
        # ts.plot()


        ts2 = timeseries.TimeSeries()
        ts2.time = np.arange(0, 10, 1)  # 10 seconds at 10 Hz
        ts2.data["Sinus"] = np.ones((10,3)) 
        ts2 = ts2.add_event(8.56, "push")
        ts2.plot("Sinus"); 
        plt.show()



        # --- advanced-----
        path = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\Sandbox\\ktk\\wheelchair_kinetics.ktk\\"
        ts = files.load(path+"wheelchair_kinetics.ktk.zip")
        # Calculate Ftot
        ts.data["Ftot"] = np.sqrt(np.sum(ts.data["Forces"] ** 2, axis=1))
        # Plot it
        ts.plot(["Forces", "Ftot"])
        
        ts_with_events = cycles.detect_cycles(
            ts,
            "Ftot",
            event_names=["push", "recovery"],
            thresholds=[10, 5],
            min_durations=[0.2, 0.2],
            min_peak_heights=[50, -np.Inf],
        )
        ts_with_events.plot(["Forces", "Ftot"])
        plt.tight_layout()


        ts_normalized_on_cycle = ktk.cycles.time_normalize(
            ts_with_events, event_name1="push", event_name2="_"
        )
        plt.figure()
        plt.subplot(2, 1, 1)
        ts_normalized_on_cycle.plot(["Forces", "Ftot"])
        plt.subplot(2, 1, 2)
        ts_normalized_on_cycle.plot("Moments")
        plt.tight_layout()

        plt.figure()
        ts_normalized_on_push = ktk.cycles.time_normalize(
            ts_with_events, "push", "recovery" )
        plt.subplot(2, 1, 1)
        ts_normalized_on_push.plot(["Forces", "Ftot"])
        plt.subplot(2, 1, 2)
        ts_normalized_on_push.plot("Moments")
        plt.tight_layout()

        plt.show()
        dataCycle = ktk.cycles.stack(ts_normalized_on_push)

        print(dataCycle)