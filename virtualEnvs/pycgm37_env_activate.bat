@echo off

rem Set the path to your Miniconda installation directory
set "CONDA_PATH=C:\Users\fleboeuf\Miniconda3"

rem Set the name of your virtual environment
set "ENV_NAME=pycgm37"

call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%