# -*- coding: utf-8 -*-

import os
import numpy as np
from bs4 import BeautifulSoup


NEXUS_PIPELINE_PATH = "C:\\Users\\Public\\Documents\\Vicon\\Nexus2.x\\Configurations\\Pipelines\\"


class NexusPipelineTests():

    @classmethod
    def Check(cls):
        for file in os.listdir(NEXUS_PIPELINE_PATH):
            if ".Pipeline" in file:
                print "------%s------"%(file)
                soup = BeautifulSoup(open(NEXUS_PIPELINE_PATH+file,"r").read(),'xml')
                params = soup.find_all("Param")
                for param in params:
                    if param.attrs["name"] == "Script":
                        print param.attrs["value"]
                        if not os.path.isfile(param.attrs["value"]):
                            raise Exception ( "Script (%s) not found"%(param.attrs["value"]))

if __name__ == "__main__":
    NexusPipelineTests.Check()
