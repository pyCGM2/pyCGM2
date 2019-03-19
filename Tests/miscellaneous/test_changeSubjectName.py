import pyCGM2
import ipdb


from pyCGM2.Tools import  btkTools

DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\changeSubjectName\\"

acq0 = btkTools.smartReader(DATA_PATH+"qualisysHug-dynamic.c3d")

btkTools.changeSubjectName(acq0,"Lecter")
btkTools.smartWriter(acq0,"testSUNBJCTNAME.c3d")
