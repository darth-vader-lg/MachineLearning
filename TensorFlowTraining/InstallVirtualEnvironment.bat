@echo off
if not exist env (
   echo Creating the Python virtual environment
   py -m venv env
)
if ERRORLEVEL 1 EXIT /B %errorlevel%
set PATH=%~dp0\env\Scripts;%~dp0;%PATH%
echo %PATH%
if not exist env\Lib\site-packages\PyInstaller (
   echo Installing the dependencies
   python -m pip install --upgrade -r requirements.txt
)
if ERRORLEVEL 1 EXIT /B %errorlevel%
if not exist env\Lib\site-packages\object_detection (
   echo Installing the object detection API
   python od_install.py
)
EXIT /B %errorlevel%
