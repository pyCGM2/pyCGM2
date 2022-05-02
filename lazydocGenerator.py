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


def generateAPI(path, module, fullmoduleName, options):

    moduleName = fullmoduleName.split(".")[-1]

    generator = MarkdownGenerator()
    markdown_docs = generator.import2md(module)

    with open(path+"_index.en.md", 'w') as f:
        print("---", file=f)
        print("title: "+moduleName, file=f)
        print("icon: \"ti-credit-card\"", file=f)
        print("description: \"Documentation API\"", file=f)
        print("type :", file=f)
        print("weight: 1", file=f)
        print("---", file=f)
        if options["Import"]:
            print("```python", file=f)
            print(
                "from " + ".".join(fullmoduleName.split(".")[0:-1]) + " import "+moduleName.split(".")[-1], file=f)
            print("```", file=f)
        print(markdown_docs, file=f)


def generateIndex():
    path = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\API\\Documentation"

    for root, dirs, filenames in os.walk(path):

        i = 0
        name = root.split("\\")[-1]

        if "_index.en.md" not in filenames:
            with open(root+"\\"+"_index.en.md", 'w') as f:
                f.writelines(["---\n",
                              "title: \"%s\"\n" % (name),
                              "icon: \"ti-credit-card\"\n",
                              "description: \"Documentation API\"\n",
                              "type :\n",
                              "weight: %s\n" % (str(i+1)),
                              "---"])


def main():

    # 1 iterer sur tous les modules
    # lire le fichier py du module
    # SI fichier py contient la ligne DOC : => creer le rep dans Doc
    # charger le module
    # generer le md

    API_PATH = "C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\Doc\\content\\API\\"
    VERSION = "Version 4.2.0"

    for (dir, subdir, filenames) in os.walk("C:\\Users\\fleboeuf\\Documents\\Programmation\\pyCGM2\\pyCGM2\\pyCGM2\\"):
        if "__init__.py" in filenames:
            pyFiles = files.getFiles(dir, extension="py")
            pyFiles.remove("__init__.py")

            for pyFile in pyFiles:

                options = {"Import": True,
                           "Draft": False,
                           "Version": VERSION}

                with open(dir+"\\"+pyFile, 'r') as f:
                    for i, line in enumerate(f):

                        if "#APIDOC[" in line:
                            key = line[line.find("[")+2:line.find("=")-2]
                            val = line[line.find("=")+1:-1]

                            if val == "True":
                                val = True
                            elif val == "False":
                                val = False
                            options.update({key: val})
                        if "#--end--" in line:
                            break

                if options != {} and "Path" in options and options["Path"]:

                    path = API_PATH+"Documentation"+"/" + options["Path"]+"/"

                    moduleName = dir[dir.rfind("pyCGM2"):].replace(
                        "\\", ".")+"."+pyFile[:-3]

                    if moduleName == "pyCGM2..enums":  moduleName = "pyCGM2.enums"

                    newpath = path+moduleName+"/"

                    mod = importlib.import_module(moduleName)

                    files.createDir(newpath)
                    generateAPI(newpath, mod, moduleName, options)

    generateIndex()


if __name__ == '__main__':
    main()
