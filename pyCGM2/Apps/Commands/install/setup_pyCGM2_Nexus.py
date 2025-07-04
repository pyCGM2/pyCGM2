import os
import sys
import subprocess
from jinja2 import Template
from pathlib import Path
import pyCGM2


# PUBLIC env may not be defined by the user
if os.getenv("PUBLIC") is not None:
    user_folder = os.getenv("PUBLIC")
else:
    user_folder = "~/"

NEXUS_PUBLIC_PATH = user_folder+"\\Documents\\Vicon\\Nexus2.x\\"
NEXUS_PUBLIC_DOCUMENT_VST_PATH = NEXUS_PUBLIC_PATH + "ModelTemplates\\"
NEXUS_PUBLIC_DOCUMENT_PIPELINE_PATH = NEXUS_PUBLIC_PATH+"Configurations\\Pipelines\\"

def find_latest_nexus_sdk():

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    vicon_dir = Path(program_files) / "Vicon"

    nexus_versions = []
    for child in vicon_dir.iterdir():
        if child.is_dir() and child.name.startswith("Nexus"):
            try:
                version = tuple(map(int, child.name.replace("Nexus", "").split(".")))
                nexus_versions.append((version, child))
            except Exception:
                continue

    if not nexus_versions:
        return None

    # On trie pour trouver la plus récente
    latest_version, latest_path = sorted(nexus_versions, reverse=True)[0]
    sdk_python_path = latest_path / "SDK" / "Win64" / "Python"

    if sdk_python_path.exists():
        return sdk_python_path
    else:
        return None

def install_vicon_packages(sdk_path):

    print(f"[pyCGM2] Détection of Vicon SDK : {sdk_path}")

    packages = ["viconnexusapi", "viconnexusutils"]
    for pkg in packages:
        wheel = sdk_path / pkg
        if wheel.exists():
            print(f"[pyCGM2]  package install : {pkg}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install -e", str(wheel)])
            except subprocess.CalledProcessError:
                print(f"[pyCGM2] ERREUR : 'install of {pkg} failed. please run from a terminal in administrator mode ")
        else:
            print(f"[pyCGM2] Package {pkg} not found in  {sdk_path}")


def main_installViconPackages():
    # Partie VICON
    sdk_path = find_latest_nexus_sdk()
    if sdk_path:
        install_vicon_packages(sdk_path)
    else:
        print("[pyCGM2] No Nexus SDK detected. Please install Vicon Nexus.")

def get_install_path(pkg):
    pkg_path = os.path.abspath(pkg.__file__)
    install_dir = os.path.dirname(pkg_path)
    parent_dir = os.path.dirname(install_dir)
    return parent_dir


def main_install_pyCGM2_NexusFiles():
    from pyCGM2.__version__ import __version__  # ou importe VERSION de manière cohérente
    
    pyversion = f"{sys.version_info.major}.{sys.version_info.minor}"
    env_name = os.environ.get("CONDA_DEFAULT_ENV", os.path.basename(sys.prefix))
    conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)

    if "envs" in conda_prefix:
        conda_root = os.path.abspath(os.path.join(conda_prefix.split("envs")[0]))
    else:
        conda_root = conda_prefix

    appdata_path = os.environ.get("APPDATA")
    target_dir = os.path.join(appdata_path, "pyCGM2")
    os.makedirs(target_dir, exist_ok=True)

    filename = f"pyCGM2-{__version__}-py{pyversion}-{env_name}-NEXUS_activate.bat"
    script_path = os.path.join(target_dir, filename)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write('rem type "conda info" to get details about minicona\n')
        f.write('@echo off\n\n')
        f.write(f'set "CONDA_PATH={conda_root}"\n')
        f.write(f'set "ENV_NAME={env_name}"\n')
        f.write('call "%CONDA_PATH%\\Scripts\\activate.bat" %ENV_NAME%\n')

    print(f"[pyCGM2] activate Script for Nexus generated : {script_path}")

    path = get_install_path(pyCGM2)

    template = path+"\\vicon\\pipeline template\\pyCGM2-CGM23-Pipeline.tpl"
    data = {
        "path": path,
        "commands_path": path+"\\Apps\\Commands\\rullThemAllCommands.py",
        "activate_path": script_path
    }
    jinja2_template_string = open(template, 'rb').read()
    template = Template(jinja2_template_string.decode("utf-8"))
    template.stream(data=data).dump(target_dir +"\\" + f"pyCGM2-{__version__}-py{pyversion}-{env_name}-CGM23.Pipeline")

    print(f"[pyCGM2] CGM23 vicon Pipeline generated : {script_path}")




main_installViconPackages()
