# Root of the workspace
workspaceRoot = ".."
# The type of the model
modelType = "SSD MobileNet v2 320x320"
# Directory where to mount the Google GDrive
mountGDriveDir = None
# ===============================================================================================================
import os
from pathlib import Path
import shutil
import subprocess
import sys

models = {
    "SSD MobileNet v2 320x320": {
        "DownloadPath": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
        "batch_size": 12
    },
}

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

# Mount the GDrive
if (mountGDriveDir != None):
    print("Mounting the GDrive")
    from google.colab import drive
    drive.mount(os.path.join(mountGDriveDir))
    workspaceGDrive = os.path.join(mountGDriveDir, "MyDrive", workspaceRoot)
    # Check the existence of the workspace directory
    if (not os.path.isdir(workspaceGDrive)):
        print("Creating the workspace")
        os.mkdir(workspaceRoot)
    os.symlink(workspaceGDrive, workspaceRoot, True)
# Check the existence of the workspace directory
else:
    if (not os.path.isdir(workspaceRoot)):
        print("Creating the workspace")

# Upgrade pip and setuptools
pythonPath = os.path.join(os.path.dirname(sys.executable), "python3")
if (not os.path.exists(pythonPath)):
    pythonPath = os.path.join(os.path.dirname(sys.executable), "python")

execute([pythonPath, "-m", "pip", "install", "--upgrade", "pip==21.0.1"])
execute([pythonPath, "-m", "pip", "install", "--upgrade", "setuptools==54.0.0"])

# Install TensorFlow
execute([pythonPath, "-m", "pip", "install", "tensorflow==2.4.1"])

# Install pygit2
execute([pythonPath, "-m", "pip", "install", "pygit2==1.5.0"])

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
            print("\r\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\rDone Deltas %d, Objects %d."%(stats.total_objects, stats.total_objects))
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
    # Move to the research dir
    currentDir = os.getcwd()
    os.chdir(os.path.join(modelsDir, "research"))
    # Install the protobuf tools
    execute([pythonPath, "-m", "pip", "install", "grpcio-tools==1.32.0"])
    # Compile the protobufs
    import grpc_tools.protoc as protoc
    protoFiles = Path("object_detection/protos").rglob("*.proto")
    for protoFile in protoFiles:
        protoFilePath = str(protoFile)
        print("Compiling", protoFilePath)
        protoc.main(["grpc_tools.protoc", "--python_out=.", protoFilePath])
    # Install the object detection packages
    shutil.copy2("object_detection/packages/tf2/setup.py", ".")
    execute([pythonPath, "-m", "pip", "install", "."])
    os.chdir(currentDir)

print("Installation completed.")

# Workspace creation
if (not os.path.isdir(os.path.join(workspaceRoot, "pre-trained-models"))):
    print("Creating the pre-trained-models directory")
    os.mkdir(os.path.join(workspaceRoot, "pre-trained-models"))
if (not os.path.isdir(os.path.join(workspaceRoot, "output-model"))):
    print("Creating the output-model directory")
    os.mkdir(os.path.join(workspaceRoot, "output-model"))

# Pre-trained model download
import urllib.request
preTrainedModelDir = str(Path(os.path.join(workspaceRoot, "pre-trained-models", Path(models[modelType]["DownloadPath"]).name)).with_suffix("").with_suffix(""))
if not (os.path.exists(preTrainedModelDir)):
    preTrainedModelFile = preTrainedModelDir + ".tar.gz"
    print(f"Downloading the pre-trained model {str(Path(preTrainedModelFile).name)}...")
    import shutil
    import tarfile
    urllib.request.urlretrieve(models[modelType]["DownloadPath"], preTrainedModelFile) # TODO: show progress
    print("Done.")
    print(f"Extracting the pre-trained model {str(Path(preTrainedModelFile).name)}...")
    tar = tarfile.open(preTrainedModelFile)
    tar.extractall(os.path.join(workspaceRoot, "pre-trained-models"))
    tar.close()
    os.remove(preTrainedModelFile)
    print("Done.")
