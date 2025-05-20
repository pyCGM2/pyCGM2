# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb


def plotCurve(ax, x,mean,sd, ofo,ofs,ifs):
    ax.plot(x , mean )
    ax.fill_between(x, mean-sd/2.0, mean+sd/2.0,
                    facecolor="blue", alpha=0.5,linewidth=0)

    ax.set_xlim(0,1)

    ax.axvline(ofo,color="black",ls='dashed')
    ax.axvline(ofs,color="black",ls='dashed')
    ax.axvline(ifs,color="black",ls='dashed')



if __name__ == "__main__":

    plt.close("all")
    xls = pd.ExcelFile("Formatted- Schwartz2008.xlsx")
    jointRotations = xls.parse("Joint Rotations")
    jointMoments = xls.parse("Joint Moments")
    jointPower = xls.parse("Joint Power")

    cycleEvents = xls.parse("Cycle Events")

    # free events
    ofo = cycleEvents[ cycleEvents["Parameter"] == "Opposite Foot Off  [% cycle]"]["FreeMean"].values[0]
    ofs = cycleEvents[ cycleEvents["Parameter"] == "Opposite Foot Contact  [% cycle]"]["FreeMean"].values[0]
    ifs = cycleEvents[ cycleEvents["Parameter"] == "Ipsilateral Foot Off  [% cycle]"]["FreeMean"].values[0]

    # ----- Kinematic -----
    data0 = jointRotations [ jointRotations["Angle"] == "Pelvic Ant/Posterior Tilt"]
    data1 = jointRotations [ jointRotations["Angle"] == "Pelvic Up/Down Obliquity"]
    data2 = jointRotations [ jointRotations["Angle"] == "Pelvic Int/External Rotation"]

    data3 = jointRotations [ jointRotations["Angle"] == "Hip Flex/Extension"]
    data4 = jointRotations [ jointRotations["Angle"] == "Hip Ad/Abduction"]
    data5 = jointRotations [ jointRotations["Angle"] == "Hip Int/External Rotation"]

    data6 = jointRotations [ jointRotations["Angle"] == "Knee Flex/Extension"]
    data7 = jointRotations [ jointRotations["Angle"] == "Knee Ad/Abduction"]
    data8 = jointRotations [ jointRotations["Angle"] == "Knee Int/External Rotation"]

    data9 = jointRotations [ jointRotations["Angle"] == "Ankle Dorsi/Plantarflexion"]

    data11 = jointRotations [ jointRotations["Angle"] == "Foot Int/External Progression"]


    fig= plt.figure()
    ax = fig.add_subplot(1,1,1)
    plotCurve(ax, data6["PercentageGaitCycle"],data6["FreeMean"],data6["FreeSd"], ofo,ofs,ifs)

    fig= plt.figure()
    ax = fig.add_subplot(1,1,1)
    plotCurve(ax, data9["PercentageGaitCycle"],data9["FreeMean"],data9["FreeSd"], ofo,ofs,ifs)

    begin = int(10/2.0+1)
    end = int(46/2.0+1)
    di = np.diff(data9["FreeMean"].values[begin:end])
    plt.figure()
    plt.plot(data9["FreeMean"].values[begin:end])
    plt.plot(di)


    slopeMax= (( data9["FreeMean"].values[end]    +data9["FreeSd"].values[end])
              - (data9["FreeMean"].values[begin]  -data9["FreeSd"].values[begin]))/(end-begin)
    slopeMin= (( data9["FreeMean"].values[end]    -data9["FreeSd"].values[end])
              - (data9["FreeMean"].values[begin]  +data9["FreeSd"].values[begin]))/(end-begin)




    # splope of dosiflexion 9 ds1 - startpushoof


    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title=u""" Descriptive Time-normalized Kinematics \n """
    fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax0 = plt.subplot(5,3,1)# Pelvis X
    ax1 = plt.subplot(5,3,2)# Pelvis Y
    ax2 = plt.subplot(5,3,3)# Pelvis Z
    ax3 = plt.subplot(5,3,4)# Hip X
    ax4 = plt.subplot(5,3,5)# Hip Y
    ax5 = plt.subplot(5,3,6)# Hip Z
    ax6 = plt.subplot(5,3,7)# Knee X
    ax7 = plt.subplot(5,3,8)# Knee Y
    ax8 = plt.subplot(5,3,9)# Knee Z
    ax9 = plt.subplot(5,3,10)# Ankle X
    ax10 = plt.subplot(5,3,11)# Ankle Z
    ax11 = plt.subplot(5,3,12)# Footprogress Z

    ax0.set_title("Pelvis Tilt" ,size=8)
    ax1.set_title("Pelvis Obliquity" ,size=8)
    ax2.set_title("Pelvis Rotation" ,size=8)
    ax3.set_title("Hip Flexion" ,size=8)
    ax4.set_title("Hip Adduction" ,size=8)
    ax5.set_title("Hip Rotation" ,size=8)
    ax6.set_title("Knee Flexion" ,size=8)
    ax7.set_title("Knee Adduction" ,size=8)
    ax8.set_title("Knee Rotation" ,size=8)
    ax9.set_title("Ankle dorsiflexion" ,size=8)
    ax10.set_title("Ankle eversion" ,size=8)
    ax11.set_title("Foot Progression " ,size=8)


    plotCurve(ax0, data0["PercentageGaitCycle"],data0["FreeMean"],data0["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax1, data1["PercentageGaitCycle"],data1["FreeMean"],data1["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax2, data2["PercentageGaitCycle"],data2["FreeMean"],data2["FreeSd"], ofo,ofs,ifs)

    plotCurve(ax3, data3["PercentageGaitCycle"],data3["FreeMean"],data3["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax4, data4["PercentageGaitCycle"],data4["FreeMean"],data4["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax5, data5["PercentageGaitCycle"],data5["FreeMean"],data5["FreeSd"], ofo,ofs,ifs)

    plotCurve(ax6, data6["PercentageGaitCycle"],data6["FreeMean"],data6["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax7, data7["PercentageGaitCycle"],data7["FreeMean"],data7["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax8, data8["PercentageGaitCycle"],data8["FreeMean"],data8["FreeSd"], ofo,ofs,ifs)

    plotCurve(ax9, data9["PercentageGaitCycle"],data9["FreeMean"],data9["FreeSd"], ofo,ofs,ifs)

    plotCurve(ax11, data11["PercentageGaitCycle"],data11["FreeMean"],data11["FreeSd"], ofo,ofs,ifs)




    # ----- Kinetics -----
    fig = plt.figure(figsize=(8.27,11.69), dpi=100,facecolor="white")
    title=u""" Descriptive Time-normalized Kinetics \n """
    fig.suptitle(title)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

    ax0 = plt.subplot(3,4,1)# Hip X extensor
    ax1 = plt.subplot(3,4,2)# Hip Y abductor
    ax2 = plt.subplot(3,4,3)# Hip Z rotation
    ax3 = plt.subplot(3,4,4)# Knee Z power

    ax4 = plt.subplot(3,4,5)# knee X extensor
    ax5 = plt.subplot(3,4,6)# knee Y abductor
    ax6 = plt.subplot(3,4,7)# knee Z rotation
    ax7 = plt.subplot(3,4,8)# knee Z power

    ax8 = plt.subplot(3,4,9)# ankle X plantar flexion
    ax9 = plt.subplot(3,4,10)# ankle Y rotation
    ax10 = plt.subplot(3,4,11)# ankle Z everter
    ax11 = plt.subplot(3,4,12)# ankle Z power

    ax0.set_title("Hip extensor Moment" ,size=8)
    ax1.set_title("Hip abductor Moment" ,size=8)
    ax2.set_title("Hip rotation Moment" ,size=8)
    ax3.set_title("Hip Power" ,size=8)

    ax4.set_title("Knee extensor Moment" ,size=8)
    ax5.set_title("Knee abductor Moment" ,size=8)
    ax6.set_title("Knee rotation Moment" ,size=8)
    ax7.set_title("Knee Power" ,size=8)

    ax8.set_title("Ankle plantarflexor Moment" ,size=8)
    ax9.set_title("Ankle everter Moment" ,size=8)
    ax10.set_title("Ankle abductor Moment" ,size=8)
    ax11.set_title("Ankle Power " ,size=8)

    data0 = jointMoments [ jointMoments["Moment"] == "Hip Ext/Flexion"]
    data1 = jointMoments [ jointMoments["Moment"] == "Hip Ab/Adduction"]
    # no Int-rot
    data3 = jointPower [ jointPower["Power"] == "Hip"]

    data4= jointMoments [ jointMoments["Moment"] == "Knee Ext/Flexion"]
    data5 = jointMoments [ jointMoments["Moment"] == "Knee Ab/Adduction"]
    # no 6
    data7 = jointPower [ jointPower["Power"] == "Knee"]

    data8 = jointMoments [ jointMoments["Moment"] == "Ankle Dorsi/Plantarflexion"]
    #no 9
    #no 10
    data11 = jointPower [ jointPower["Power"] == "Ankle"]


    plotCurve(ax0, data0["PercentageGaitCycle"],data0["FreeMean"],data0["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax1, data1["PercentageGaitCycle"],data1["FreeMean"],data1["FreeSd"], ofo,ofs,ifs)
    # no2
    plotCurve(ax3, data3["PercentageGaitCycle"],data3["FreeMean"],data3["FreeSd"], ofo,ofs,ifs)

    plotCurve(ax4, data4["PercentageGaitCycle"],data4["FreeMean"],data4["FreeSd"], ofo,ofs,ifs)
    plotCurve(ax5, data5["PercentageGaitCycle"],data5["FreeMean"],data5["FreeSd"], ofo,ofs,ifs)
    # no6
    plotCurve(ax7, data7["PercentageGaitCycle"],data7["FreeMean"],data7["FreeSd"], ofo,ofs,ifs)

    plotCurve(ax8, data8["PercentageGaitCycle"],data8["FreeMean"],data8["FreeSd"], ofo,ofs,ifs)
    # no9
    # no 10
    plotCurve(ax11, data11["PercentageGaitCycle"],data11["FreeMean"],data11["FreeSd"], ofo,ofs,ifs)




    plt.show()
