@echo off
call InstallVirtualEnvironment.bat
if ERRORLEVEL 1 EXIT /B %errorlevel%
echo Creating the executable
pyinstaller --onefile --clean ^
   --hidden-import pandas._libs.tslibs.base ^
   --hidden-import tensorflow.python.keras.engine.base_layer_v1 ^
   --add-binary .\env\Scripts\tensorboard.exe;. ^
   --runtime-tmpdir %TEMP%\tf-od-model-builder ^
   --distpath .\tf-od-model-builder ^
   -n tf-od-model-builder ^
   main.py
EXIT /B %errorlevel%
