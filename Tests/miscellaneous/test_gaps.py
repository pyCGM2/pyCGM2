import pyCGM2
import matplotlib.pyplot as plt


from pyCGM2.Tools import  btkTools

DATA_PATH = pyCGM2.TEST_DATA_PATH+"operations\\gapFilling\\checkResiduals\\"

raw = btkTools.smartReader(DATA_PATH+"PN01OP01S01SS01-raw.c3d")
tiapRaw = raw.GetPoint("LTIAP")

spline = btkTools.smartReader(DATA_PATH+"PN01OP01S01SS01-spline.c3d")
tiapSpline = spline.GetPoint("LTIAP")


pick = btkTools.smartReader(DATA_PATH+"PN01OP01S01SS01-pick.c3d")
tiapPick = pick.GetPoint("LTIAP")

rigid = btkTools.smartReader(DATA_PATH+"PN01OP01S01SS01-rigid.c3d")
tiapRigid = rigid.GetPoint("LTIAP")


plt.plot(tiapRaw.GetResiduals())
plt.plot(tiapSpline.GetResiduals())
plt.plot(tiapPick.GetResiduals())
plt.plot(tiapRigid.GetResiduals())
plt.show()
