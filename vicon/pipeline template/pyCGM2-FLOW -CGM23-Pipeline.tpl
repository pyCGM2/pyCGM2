<?xml version="1.1" encoding="UTF-8" standalone="no" ?>
<Pipeline>

   <Entry DisplayName="pyCGM2-FLOW-Edit" Enabled="0" OperationId="0" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="FLOW Edit -cgm CGM2.3 -d"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-FLOW-Prepare" Enabled="0" OperationId="0" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="FLOW Prepare -u  CGM23_v2"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-FLOW-MEK-Populate" Enabled="0" OperationId="0" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="FLOW Populate -u CGM23_v2  --update"/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

  <Entry DisplayName="pyCGM2-FLOW-Report" Enabled="0" OperationId="0" OperationName="Python">
    <ParamList name="" version="1">
      <Param name="Script" value="{{ data.commands_path }}"/>
      <Param name="ScriptArgs" value="FLOW Report -u GM23_v2 "/>
      <Param name="PythonCommand" value="python.exe"/>
      <Param name="PythonPreScript" value="{{ data.activate_path }}"/>
      <Param name="LaunchPython" value="false"/>
      <Param name="UseNexusPython" value="false"/>
    </ParamList>
  </Entry>

</Pipeline>
