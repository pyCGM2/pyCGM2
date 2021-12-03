from lazydocs import MarkdownGenerator


def generate(DATA_PATH, filenameNoExt, module):
    generator = MarkdownGenerator()

    markdown_docs = generator.import2md(module)

    with open(DATA_PATH+filenameNoExt+".md", 'w') as f:
        print("---", file=f)
        print("title: "+filenameNoExt, file=f)
        print("---", file=f)
        print(markdown_docs, file=f)


def main():

    PATH = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\\API\\Version 4.2\\Lib\\"
    from pyCGM2.Lib import emg
    generate(PATH, "pyCGM2.Lib.emg", emg)

    from pyCGM2.Lib import analysis
    generate(PATH, "pyCGM2.Lib.analysis", analysis)

    from pyCGM2.Lib import eventDetector
    generate(PATH, "pyCGM2.Lib.eventDetector", eventDetector)

    from pyCGM2.Lib import plot
    generate(PATH, "pyCGM2.Lib.plot", plot)

    from pyCGM2.Lib.CGM import cgm1
    generate(PATH, "pyCGM2.Lib.CGM.cgm1", cgm1)

    from pyCGM2.Lib.CGM import cgm1_1
    generate(PATH, "pyCGM2.Lib.CGM.cgm1_1", cgm1_1)

    from pyCGM2.Lib.CGM import cgm2_1
    generate(PATH, "pyCGM2.Lib.CGM.cgm2_1", cgm2_1)

    from pyCGM2.Lib.CGM import cgm2_2
    generate(PATH, "pyCGM2.Lib.CGM.cgm2_2", cgm2_2)

    from pyCGM2.Lib.CGM import cgm2_3
    generate(PATH, "pyCGM2.Lib.CGM.cgm2_3", cgm2_3)

    from pyCGM2.Lib.CGM import cgm2_4
    generate(PATH, "pyCGM2.Lib.CGM.cgm2_4", cgm2_4)

    from pyCGM2.Lib.CGM import cgm2_5
    generate(PATH, "pyCGM2.Lib.CGM.cgm2_5", cgm2_5)

    PATH = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\\API\\Version 4.2\\Apps\\Commands\\"
    from pyCGM2.Apps.Commands import commands
    generate(PATH, "pyCGM2.Apps.Commands", commands)

    PATH = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\\API\\Version 4.2\\Apps\\Vicon\\"
    from pyCGM2.Apps.ViconApps.CGM1 import CGM1_Calibration, CGM1_Fitting
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM1.Calibration", CGM1_Calibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM1.Fitting", CGM1_Fitting)

    from pyCGM2.Apps.ViconApps.CGM1_1 import CGM1_1_Calibration, CGM1_1_Fitting
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM1_1.Calibration", CGM1_1_Calibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM1_1.Fitting", CGM1_1_Fitting)

    from pyCGM2.Apps.ViconApps.CGM2_1 import CGM2_1_Calibration, CGM2_1_Fitting
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_1.Calibration", CGM2_1_Calibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_1.Fitting", CGM2_1_Fitting)

    from pyCGM2.Apps.ViconApps.CGM2_2 import CGM2_2_Calibration, CGM2_2_Fitting
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_2.Calibration", CGM2_2_Calibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_2.Fitting", CGM2_2_Fitting)

    from pyCGM2.Apps.ViconApps.CGM2_3 import CGM2_3_Calibration, CGM2_3_Fitting
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_3.Calibration", CGM2_3_Calibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_3.Fitting", CGM2_3_Fitting)

    from pyCGM2.Apps.ViconApps.CGM2_4 import CGM2_4_Calibration, CGM2_4_Fitting
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_4.Calibration", CGM2_4_Calibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_4.Fitting", CGM2_4_Fitting)

    from pyCGM2.Apps.ViconApps.CGM2_5 import CGM2_5_Calibration, CGM2_5_Fitting
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_5.Calibration", CGM2_5_Calibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_5.Fitting", CGM2_5_Fitting)

    from pyCGM2.Apps.ViconApps.CGM2_6 import CGM_Knee2DofCalibration, CGM_KneeSARA
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_6.Calibration2dof", CGM_Knee2DofCalibration)
    generate(PATH, "pyCGM2.Apps.ViconApps.CGM2_6.Sara", CGM_KneeSARA)

if __name__ == '__main__':
    main()
