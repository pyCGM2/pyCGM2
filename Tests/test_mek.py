# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_moveckAPI.py
import sys

from pyCGM2.Mek import mekScheme
from pyCGM2.Mek import mekExtract
from pyCGM2.Mek import mekNormalize

MOVECKPATH = "C:\\Users\\fleboeuf\\Documents\\2. AREA OF RESPONSABILITY\\Programmation\\moveck\\"
sys.path.append(MOVECKPATH+"Moveck_pipe-2024.1.0-win64-pipeline_install\\packages")
import moveck





class GechoSchemeProcedure(mekScheme.AbstractSchemeProcedure):
    def __init__(self):
        super(GechoSchemeProcedure,self).__init__()
        self.Kinematics =  []
        self.Kinetics =  []
        self.EMG =  []
        self.MTJ =  []



def main():
    path = "C:\\Users\\fleboeuf\\Documents\\DATA\\pyCGM2-Data-Tests\\mek\\gaitdata\\"
    filenames = ["gait Trial 01.c3d","gait Trial 02.c3d"]

    ds = moveck.data_store()
    root = ds.root()


    scheme = GechoSchemeProcedure()
    scheme.setData("Kinematics", [path+filename for filename in filenames], targets = ["LHipAngles.Left","RHipAngles.Right"])


    filter = mekExtract.mekStorageFilter(ds)
    filter.run(scheme)


    normalize_filter = mekNormalize.mekNormalizeFilter(ds)
    normalize_filter.run(scheme)  

    ds.dump("storage.h5")


if __name__ == '__main__':
    main()



