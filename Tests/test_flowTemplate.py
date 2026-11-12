# coding: utf-8
# pytest -s --disable-pytest-warnings test_flowTemplate.py
import os
import pytest

import pyCGM2
from pyCGM2.flow import flowFilters

TEMPLATE_CONTENT = "ModelVersion: {{ data.ModelVersion }}\n"


def _writeTemplate(folder, content=TEMPLATE_CONTENT):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, flowFilters.FLOW_TEMPLATE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


@pytest.fixture
def isolatedLocations(tmp_path, monkeypatch):
    """No environment variable, empty APPDATA and empty pyCGM2 settings folder."""
    monkeypatch.delenv(flowFilters.FLOW_TEMPLATE_ENV_VARIABLE, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(pyCGM2, "PYCGM2_SETTINGS_FOLDER", str(tmp_path / "settings") + os.sep)
    return tmp_path


class Test_findFlowTemplate:

    def test_notFound(self, isolatedLocations):
        with pytest.raises(FileNotFoundError, match="--template"):
            flowFilters.findFlowTemplate()

    def test_settingsFolder(self, isolatedLocations):
        expected = _writeTemplate(str(isolatedLocations / "settings" / "templates"))
        assert flowFilters.findFlowTemplate() == expected

    def test_userFolderBeforeSettingsFolder(self, isolatedLocations):
        _writeTemplate(str(isolatedLocations / "settings" / "templates"))
        expected = _writeTemplate(os.path.join(str(isolatedLocations / "appdata"), "pyCGM2", "templates"))
        assert flowFilters.findFlowTemplate() == expected

    def test_envVariableBeforeUserFolder(self, isolatedLocations, monkeypatch):
        _writeTemplate(os.path.join(str(isolatedLocations / "appdata"), "pyCGM2", "templates"))
        expected = _writeTemplate(str(isolatedLocations / "custom"))
        monkeypatch.setenv(flowFilters.FLOW_TEMPLATE_ENV_VARIABLE, expected)
        assert flowFilters.findFlowTemplate() == expected

    def test_argumentBeforeEnvVariable(self, isolatedLocations, monkeypatch):
        monkeypatch.setenv(flowFilters.FLOW_TEMPLATE_ENV_VARIABLE, _writeTemplate(str(isolatedLocations / "custom")))
        expected = _writeTemplate(str(isolatedLocations / "argument"))
        assert flowFilters.findFlowTemplate(expected) == expected

    def test_missingExplicitPathDoesNotFallBack(self, isolatedLocations, monkeypatch):
        _writeTemplate(os.path.join(str(isolatedLocations / "appdata"), "pyCGM2", "templates"))
        with pytest.raises(FileNotFoundError):
            flowFilters.findFlowTemplate(str(isolatedLocations / "missing.tpl"))
        monkeypatch.setenv(flowFilters.FLOW_TEMPLATE_ENV_VARIABLE, str(isolatedLocations / "missing.tpl"))
        with pytest.raises(FileNotFoundError, match=flowFilters.FLOW_TEMPLATE_ENV_VARIABLE):
            flowFilters.findFlowTemplate()


class Test_FlowEdittingFilterTemplate:

    def test_runWithUserFolderTemplate(self, isolatedLocations):
        _writeTemplate(os.path.join(str(isolatedLocations / "appdata"), "pyCGM2", "templates"))
        data_path = str(isolatedLocations / "session") + os.sep
        os.makedirs(data_path)

        fef = flowFilters.FlowEdittingFilter(data_path, "CGM2.3", emgSettings={"CHANNELS": {}})
        fef.run("CGM23_v2.settings", displayFile=False)

        with open(data_path + "CGM23_v2.settings", encoding="utf-8") as f:
            assert f.read().strip() == "ModelVersion: CGM2.3"

    def test_runWithSetTemplate(self, isolatedLocations):
        template = _writeTemplate(str(isolatedLocations / "custom"), "custom {{ data.ModelVersion }}")
        data_path = str(isolatedLocations / "session") + os.sep
        os.makedirs(data_path)

        fef = flowFilters.FlowEdittingFilter(data_path, "CGM2.1", emgSettings={"CHANNELS": {}})
        fef.setTemplate(template)
        fef.run("CGM21_v2.settings", displayFile=False)

        with open(data_path + "CGM21_v2.settings", encoding="utf-8") as f:
            assert f.read().strip() == "custom CGM2.1"
