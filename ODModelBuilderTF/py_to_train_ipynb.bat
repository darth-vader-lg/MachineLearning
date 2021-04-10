@echo off
rem Updating of the train.ipynb with the content of the python modules
setlocal
set SyncJupyterNotebook="..\SyncJupyterNotebook\SyncJupyterNotebook\bin\Release\net5.0\SyncJupyterNotebook.exe"
if not exist %SyncJupyterNotebook% (
  set SyncJupyterNotebook="..\SyncJupyterNotebook\SyncJupyterNotebook\bin\Debug\net5.0\SyncJupyterNotebook.exe"
  if not exist %SyncJupyterNotebook% (
    echo the SyncJupyterNotebook executable doesn't exist. Please build the solution.
    exit 1
  )
)

%SyncJupyterNotebook% nb train.ipynb
