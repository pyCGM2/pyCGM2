import os
import subprocess
from pathlib import Path
from jinja2 import Environment, BaseLoader
from pyCGM2.Utils import files
from pyCGM2.flow import settingsHandler


TEMPLATE = """Trials:
{% for trial in trials %}  - Name : {{ trial.name }}
    MeasurementType : {{ trial.measurement_type }}
    Filename : {{ trial.filename }}
{% if not loop.last %}
{% endif %}{% endfor %}
"""
def build_trial(data_path,filename,type):
            return {
                "name": filename,
                "measurement_type": type,
                "filename": data_path+"/"+filename,
            }


class PipeSessionSingleFileProcedure(object):

    def __init__(self, trial_name, type="Dynamic"):
        self.m_trial_name = trial_name
        self.m_type = type

    def run(self, data_path):
        trial = [build_trial(data_path, self.m_trial_name, self.m_type)]

        env = Environment(loader=BaseLoader(), trim_blocks=False, lstrip_blocks=False)
        render= env.from_string(TEMPLATE).render(trials=trial)

        return render




class PipeSessionFlowUserSettingsProcedure(object):

    def __init__(self, user_settings):
        self.m_user_settings = user_settings

    def run(self, data_path):

        


        calibrationTrialNames = settingsHandler.get_CalibrationTrials(self.m_user_settings)
        fittingTrialNames = settingsHandler.get_FittingTrials(self.m_user_settings)


        staticTrials = [build_trial(data_path, filename, "Static") for filename in calibrationTrialNames]
        fittingTrials = [build_trial(data_path, filename, "Dynamic") for filename in fittingTrialNames]

        trials = staticTrials + fittingTrials


        env = Environment(loader=BaseLoader(), trim_blocks=False, lstrip_blocks=False)
        render= env.from_string(TEMPLATE).render(trials=trials)

    
        return render



class PipeSessionBuilderFilter(object):
    """Classe pour filtrer les essais dans un fichier YAML de session Pipe."""


    def __init__(self, procedure):
        self.m_procedure = procedure


    def run(self,data_path):
        render = self.m_procedure.run(data_path)
    
        files.createDir(data_path+"pipe")

        if isinstance(self.m_procedure,PipeSessionFlowUserSettingsProcedure):
            sessionPathFile = data_path+"pipe\\pipe_flow_session.yaml"
        else:
            sessionPathFile = data_path+"pipe\\_session.yaml"

        Path(sessionPathFile).write_text(render, encoding="utf-8") 







PIPE_EXE = os.getenv("PIPE_EXE")

def run_pipe(
    environment,
    dependencies_folder,
    pipeline,
    destination,
    session,
    source,
    overwrite=True,
    quiet=False,
    pipe_exe=PIPE_EXE):

    """Construit et exécute la commande Pipe.
    result = run_pipe(
                        environment=MOVECK_BASE / f"{envFile}",
                        dependencies_folder=MOVECK_BASE / "pipelines/",
                        pipeline=pipelineName,
                        destination= f"{main_path}/{output_h5file}",
                        session=f"{main_path}/{output_file}",  # = le YAML généré
                        source=session_path,
                        overwrite=True,
                        quiet=True,
                    )
    """



    def cmd_list_to_string(cmd):
        """Convertit une liste subprocess en chaîne copiable dans une console DOS."""
        parts = []
        for arg in cmd:
            # On entoure de guillemets tout argument contenant un espace
            if " " in arg:
                parts.append(f'"{arg}"')
            else:
                parts.append(arg)
        return " ".join(parts)


    

    cmd = [
        pipe_exe,
        "-e", str(environment),
        "-f", str(dependencies_folder),
        "-p", pipeline,
        "-d", str(destination),
        "-i", str(session),
        "-r", str(source),
    ]
    if overwrite:
        cmd.append("-o")
    if quiet:
        cmd.append("-q")


    out = cmd_list_to_string(cmd)  # Convert command list to string for debugging


    result = subprocess.run(cmd, capture_output=True, text=True)
    return result