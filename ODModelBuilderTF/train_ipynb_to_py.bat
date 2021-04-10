@echo off
rem Updating of the python modules with the content of the train.ipynb
setlocal
set SyncJupyterNotebook="..\SyncJupyterNotebook\SyncJupyterNotebook\bin\Release\net5.0\SyncJupyterNotebook.exe"
if not exist %SyncJupyterNotebook% (
  set SyncJupyterNotebook="..\SyncJupyterNotebook\SyncJupyterNotebook\bin\Debug\net5.0\SyncJupyterNotebook.exe"
  if not exist %SyncJupyterNotebook% (
    echo the SyncJupyterNotebook executable doesn't exist. Please build the solution.
    exit 1
  )
)

%SyncJupyterNotebook% py train.ipynb
