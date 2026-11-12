# -*- coding: utf-8 -*-
"""Filters of the *flow* pipeline."""
import os
import pyCGM2
LOGGER = pyCGM2.LOGGER
from jinja2 import Template
from pyCGM2.flow import settingsHandler
from pyCGM2.Utils import files

FLOW_TEMPLATE_FILENAME = "cgm#i-settings.tpl"
FLOW_TEMPLATE_ENV_VARIABLE = "PYCGM2_FLOW_TEMPLATE"


def getUserTemplateFolder():
    """Return the user folder of the pyCGM2 templates (``%APPDATA%/pyCGM2/templates``).

    Returns:
        Optional[str]: Folder path, or None if the ``APPDATA`` environment variable is not defined.
    """
    appdata = os.getenv("APPDATA")
    return os.path.join(appdata, "pyCGM2", "templates") if appdata is not None else None


def findFlowTemplate(template_pathFilename=None):
    """Find the template of the flow user settings file.

    The template is not distributed with pyCGM2. It is searched in this order:

    1. ``template_pathFilename``, if provided;
    2. the path given by the ``PYCGM2_FLOW_TEMPLATE`` environment variable (it can be set in a ``.env`` file);
    3. the user folder ``%APPDATA%/pyCGM2/templates/cgm#i-settings.tpl``;
    4. the pyCGM2 settings folder ``Settings/templates/cgm#i-settings.tpl`` (development copy).

    An explicit path (1 or 2) which does not exist raises an error, without falling back to the other locations.

    Args:
        template_pathFilename (str, optional): Path of the template.

    Returns:
        str: Path of the template.

    Raises:
        FileNotFoundError: If the template is not found.
    """
    if template_pathFilename is not None:
        if not os.path.isfile(template_pathFilename):
            raise FileNotFoundError(f"[pyCGM2] flow template not found: {template_pathFilename}")
        return template_pathFilename

    envTemplate = os.getenv(FLOW_TEMPLATE_ENV_VARIABLE)
    if envTemplate:
        if not os.path.isfile(envTemplate):
            raise FileNotFoundError(
                f"[pyCGM2] flow template not found: {envTemplate} (defined by the {FLOW_TEMPLATE_ENV_VARIABLE} environment variable)")
        return envTemplate

    candidates = []
    userFolder = getUserTemplateFolder()
    if userFolder is not None:
        candidates.append(os.path.join(userFolder, FLOW_TEMPLATE_FILENAME))
    candidates.append(os.path.join(pyCGM2.PYCGM2_SETTINGS_FOLDER, "templates", FLOW_TEMPLATE_FILENAME))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"[pyCGM2] flow template ({FLOW_TEMPLATE_FILENAME}) not found. Copy it into "
        f"{userFolder if userFolder is not None else '%APPDATA%/pyCGM2/templates'}, set the "
        f"{FLOW_TEMPLATE_ENV_VARIABLE} environment variable, or pass its path with --template. "
        f"Searched: {', '.join(candidates)}")


class FlowEdittingFilter(object):
    """Filter writing the flow user settings file (``cgm#i-settings``) of a session.

    The template of the settings file is located with :func:`findFlowTemplate`, unless
    it is set with :meth:`setTemplate`.

    Args:
        data_path (str): Session folder.
        modelVersion (str): CGM version.
        procedure (pyCGM2.flow.procedures.flowProcedures.AbstractFlowProcedure, optional): Procedure
            collecting session details (e.g. from Eclipse). If None, an empty settings structure is written.
        emgSettings (dict, optional): EMG settings. Defaults to the pyCGM2 ``emg.settings`` file.
    """

    def __init__(self,data_path, modelVersion,procedure=None, emgSettings = None):
        self.m_template = None

        self.m_procedure = procedure

        self.m_modelVersion = modelVersion
        self.m_data_path = data_path

        if emgSettings is  None:

            emgSettings = files.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,"emg.settings")

            self.m_emg_list = settingsHandler.emg_ordered_dict_to_list(emgSettings["CHANNELS"])

        else:
            self.m_emg_list = settingsHandler.emg_ordered_dict_to_list(emgSettings["CHANNELS"])



    def setTemplate(self, template_pathFilename):
        """Set the template used to render the settings file.

        Args:
            template_pathFilename (str): Path of the template. If None, the template is located
                with :func:`findFlowTemplate`.
        """
        self.m_template = template_pathFilename


    def run(self, filename, displayFile=True):
        """Render and write the settings file.

        Args:
            filename (str): Output filename.
            displayFile (bool, optional): Open the generated file. Defaults to True.

        Raises:
            FileNotFoundError: If the template is not found.
        """
        templatePathFilename = findFlowTemplate(self.m_template)
        LOGGER.logger.info(f"[pyCGM2] flow template: {templatePathFilename}")

        if self.m_procedure is not None:
            data = self.m_procedure.run(self.m_data_path,self.m_modelVersion)
            data["Emg_settings"] = self.m_emg_list

        else: 
            data = {
                "ModelVersion": self.m_modelVersion,
                "Patient": {},
                "Visit": {},
                "Mp": {},
                "Calibration": [],
                "Fitting": [],
                "Emg": [],
                "Conditions": [],
                "Emg_settings": []
            }
        if self.m_data_path is None : self.m_data_path=""

        jinja2_template_string = open(templatePathFilename, 'rb').read()
        template = Template(jinja2_template_string.decode("utf-8"))
        template.stream(data=data).dump(self.m_data_path+filename)

        if  displayFile :  os.startfile(self.m_data_path+filename)
