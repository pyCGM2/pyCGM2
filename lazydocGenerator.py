import pkgutil
import os
import pyCGM2
from lazydocs import MarkdownGenerator
from pyCGM2.Utils import files
import importlib


API_PATH = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\\API\\"
VERSION = "Version 4.2"


modules = list()
for imp, mod, isp in pkgutil.walk_packages(path=pyCGM2.__path__, prefix=pyCGM2.__name__+"."):
    modules.append(mod)


def en_index(title, weight):
    content = """
    ---
    title: "%s"
    #date: 2018-12-28T11:02:05+06:00
    icon: "ti-credit-card" # themify icon pack : https://themify.me/themify-icons
    description: "Documentation API  "
    # type dont remove or customize
    type : "docs"
    weight: %s
    ---
    """ % (title, weight)

    return content


# def createDir(fullPathName):
#     fullPathName = fullPathName
#     pathOut = fullPathName[:-1] if fullPathName[-1:]=="\\" else fullPathName
#     if not os.path.isdir((pathOut)):
#         os.makedirs((pathOut))
#     else:
#         LOGGER.logger.info("directory already exists")


def generate(DATA_PATH, filenameNoExt, module):

    files.createDir(DATA_PATH)

    generator = MarkdownGenerator()

    markdown_docs = generator.import2md(module)

    with open(DATA_PATH+filenameNoExt+".md", 'w') as f:
        print("---", file=f)
        print("title: "+filenameNoExt, file=f)
        print("---", file=f)
        print(markdown_docs, file=f)


def main():

    # 1 iterer sur tous les modules
    # lire le fichier py du module
    # SI fichier py contient la ligne DOC : => creer le rep dans Doc
    # charger le module
    # generer le md

    API_PATH = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\\API\\"
    VERSION = "Version 4.3"

    for (dir, subdir, filenames) in os.walk("C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\"):
        if "__init__.py" in filenames:
            pyFiles = files.getFiles(dir, extension="py")
            pyFiles.remove("__init__.py")

            for pyFile in pyFiles:

                flag = False
                with open(dir+"\\"+pyFile, 'r') as f:
                    for i, line in enumerate(f):
                        if "#APIDOC:" in line:
                            pathInApiDoc = line.split(
                                ":")[1][:-1][1:]  # .replace(" ", "")
                            flag = True
                            break
                if flag:
                    newpath = API_PATH+VERSION+"/"+pathInApiDoc+"/"
                    files.createDir(newpath)

                    moduleName = dir[dir.rfind("pyCGM2"):].replace(
                        "\\", ".")+"."+pyFile[:-3]
                    mod = importlib.import_module(moduleName)

                    generate(newpath, dir[dir.rfind("pyCGM2"):].replace(
                        "\\", ".")+"."+pyFile[:-3], mod)


if __name__ == '__main__':
    main()
