@echo off
if not exist env (
   echo Creating the Python virtual environment
   py -m venv env
)
if ERRORLEVEL 1 EXIT /B %errorlevel%
call env\Scripts\activate.bat
if not exist env\Lib\site-packages\PyInstaller (
   echo Installing the dependencies
   py -m pip install -r requirements.txt
)
if ERRORLEVEL 1 EXIT /B %errorlevel%
if not exist env\Lib\site-packages\object_detection (
   echo Installing the object detection API
   py od_install.py
)
EXIT /B %errorlevel%
