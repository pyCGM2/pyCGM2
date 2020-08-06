# -*- coding: utf-8 -*-
# this file is an alternative to typical pytest process.
# I actually came across an issue with pytest when
# i want to run both test_CGM1.py and test_CGM1_Kinetics. pytest block and exit
# at the last method of test_CGM1_Kinetics.py....don't know why :-(
# there is apparently an compatibility between both tests. ( memory leak...?)




import pyCGM2
import os

def getTestFiles(path):
    pyfiles = list()
    for dirpath, dirs, files in os.walk(path):
      for filename in files:
        # fname = os.path.join(dirpath,filename)
        fname = filename
        if fname.endswith('.py'):
            pyfiles.append(fname)
    return pyfiles


pytestfiles = getTestFiles(pyCGM2.MAIN_PYCGM2_PATH+"Tests")
for testfile in pytestfiles:
    if "plot" in testfile:
        cmd = "pytest -v --disable-pytest-warnings --mpl --exitfirst Tests/" + testfile
    else:
        cmd = "pytest -v --disable-pytest-warnings --exitfirst Tests/" + testfile

    val = os.system(cmd)

    if val == 1: raise Exception("File %s fails"%(testfile) )




if val==0:  print "GREAT.....ALL TESTS PASSED......:-) "
