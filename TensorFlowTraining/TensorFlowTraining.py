# Root of the workspace
workspaceRoot = ".."
# ===============================================================================================================
import os
from pathlib import Path
import shutil
import subprocess
import sys

# Execute a subprocess
def executeSubprocess(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

# Execute with output
def execute(cmd):
    for output in executeSubprocess(cmd):
        print(output, end="")

# Upgrade pip and setuptools
pythonPath = os.path.join(os.path.dirname(sys.executable), "python3")
if (not os.path.exists(pythonPath)):
    pythonPath = os.path.join(os.path.dirname(sys.executable), "python")

execute([pythonPath, "-m", "pip", "install", "--upgrade", "pip"])
execute([pythonPath, "-m", "pip", "install", "--upgrade", "setuptools"])

# Install TensorFlow
execute([pythonPath, "-m", "pip", "install", "tensorflow==2.4.1"])

# Install pygit2
execute([pythonPath, "-m", "pip", "install", "pygit2"])

# Progress for git
import pygit2
import datetime
class GitCallbacks(pygit2.RemoteCallbacks):
    def __init__(self, credentials=None, certificate=None):
        self.dateTime = datetime.datetime.now()
        return super().__init__(credentials=credentials, certificate=certificate)
    def transfer_progress(self, stats):
        now = datetime.datetime.now()
        if ((now - self.dateTime).total_seconds() > 1):
            print("\rReceiving... Deltas [%d / %d], Objects [%d / %d]"%(stats.indexed_deltas, stats.total_deltas, stats.indexed_objects, stats.total_objects), end="", flush=True)
            self.dateTime = now
        if (stats.received_objects >= stats.total_objects and stats.indexed_objects >= stats.total_objects and stats.indexed_deltas >= stats.total_deltas):
            print("\rDone Deltas %d, Objects %d."%(stats.total_objects, stats.total_objects))
        return super().transfer_progress(stats)

# Directory of the TensorFlow models
modelsDir = os.path.join(workspaceRoot, "models")
# Install the TensorFlow models
if (not os.path.isdir(modelsDir)):
    # Create the callback for the progress
    callbacks = GitCallbacks();
    # Clone the TensorFlow models repository
    print("Cloning the TensorFlow models repository")
    pygit2.clone_repository("https://github.com/tensorflow/models.git", modelsDir, callbacks = callbacks)
    print("TensorFlow models repository cloned")
    # Checkout a well known commit
    repo = pygit2.Repository(modelsDir)
    ish = "e356598a5b79a768942168b10d9c1acaa923bdb4"
    (commit, reference) = repo.resolve_refish(ish)
    repo.checkout_tree(commit)
    repo.reset(pygit2.Oid(hex=ish), pygit2.GIT_RESET_HARD)
    # Install the object detection packages
    currentDir = os.getcwd()
    os.chdir(os.path.join(modelsDir, "research"))
    shutil.copy2("object_detection/packages/tf2/setup.py", ".")
    execute([pythonPath, "-m", "pip", "install", "."])
    # Install the protobuf tools
    execute([pythonPath, "-m", "pip", "install", "grpcio-tools"])
    # Compile the protobufs
    import grpc_tools.protoc as protoc
    protoFiles = Path("object_detection/protos").rglob("*.proto")
    for protoFile in protoFiles:
        protoFilePath = str(protoFile)
        print("Compiling", protoFilePath)
        protoc.main(["grpc_tools.protoc", "--python_out=.", protoFilePath])
    os.chdir(currentDir)

print("Installation completed.")
