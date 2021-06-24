from lazydocs import MarkdownGenerator

def generate(DATA_PATH,filenameNoExt,module):
    generator = MarkdownGenerator()

    markdown_docs = generator.import2md(module)

    with open(DATA_PATH+filenameNoExt+".md", 'w') as f:
        print("---", file=f)
        print("title: "+filenameNoExt, file=f)
        print("---", file=f)
        print(markdown_docs,file=f)

def main():

    PATH = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\\API\\Version 4.1.1\\Lib\\"
    from pyCGM2.Lib import emg
    generate(PATH,"pyCGM2.Lib.emg",emg)

    from pyCGM2.Lib import analysis
    generate(PATH,"pyCGM2.Lib.analysis",analysis)


    from pyCGM2.Lib import eventDetector
    generate(PATH,"pyCGM2.Lib.eventDetector",eventDetector)


    from pyCGM2.Lib import plot
    generate(PATH,"pyCGM2.Lib.plot",plot)


    from pyCGM2.Lib.CGM import cgm1
    generate(PATH,"pyCGM2.Lib.CGM.cgm1",cgm1)

    from pyCGM2.Lib.CGM import cgm1_1
    generate(PATH,"pyCGM2.Lib.CGM.cgm1_1",cgm1_1)

    from pyCGM2.Lib.CGM import cgm2_1
    generate(PATH,"pyCGM2.Lib.CGM.cgm2_1",cgm2_1)

    from pyCGM2.Lib.CGM import cgm2_2
    generate(PATH,"pyCGM2.Lib.CGM.cgm2_2",cgm2_2)

    from pyCGM2.Lib.CGM import cgm2_3
    generate(PATH,"pyCGM2.Lib.CGM.cgm2_3",cgm2_3)

    from pyCGM2.Lib.CGM import cgm2_4
    generate(PATH,"pyCGM2.Lib.CGM.cgm2_4",cgm2_4)

    from pyCGM2.Lib.CGM import cgm2_5
    generate(PATH,"pyCGM2.Lib.CGM.cgm2_5",cgm2_5)



if __name__ == '__main__':
    main()
