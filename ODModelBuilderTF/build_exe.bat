@echo off
call InstallVirtualEnvironment.bat
if ERRORLEVEL 1 EXIT /B %errorlevel%
echo Creating the executable
pyinstaller --onefile --clean ^
   --hidden-import pandas._libs.tslibs.base ^
   --hidden-import tensorflow.python.keras.engine.base_layer_v1 ^
   --add-binary .\env\Scripts\tensorboard.exe;. ^
   --runtime-tmpdir %TEMP%\tf-od-model-builder ^
   --distpath .\ODModelBuilderTF ^
   -n ODModelBuilderTF ^
   main.py
if ERRORLEVEL 1 EXIT /B %errorlevel%
echo cleaning
rd /S /Q build
EXIT /B %errorlevel%
