# Module od_install.py
#@title #Object detection libraries installation { form-width: "20%" }
#@markdown This step installs a well known Python environment for the train.

import  os
import  datetime
from    pathlib import Path
import  shutil
import  sys
import  tempfile

try:    from utilities import *
except: pass

def install_object_detection():
    """
    Install a well known environment.
    """
    # Upgrade pip and setuptools
    is_installed = False
    try:
        import pip
        is_installed = pip.__version__ == '21.0.1'
    except: pass
    if (not is_installed):
        execute_script(['-m', 'pip', 'install', '--upgrade', 'pip==21.0.1'])
    else:
        print('pip 21.0.1 is already installed')
    is_installed = False
    try:
        import setuptools
        is_installed = setuptools.__version__ == '54.0.0'
    except: pass
    if (not is_installed):
        execute_script(['-m', 'pip', 'install', '--upgrade', 'setuptools==54.0.0'])
    else:
        print('setuptools 54.0.0 is already installed')
    # Install TensorFlow
    is_installed = False
    tensorflow_version = 'tensorflow==2.4.1' # or for example tf-nightly==2.5.0.dev20210315
    try:
        import tensorflow
        comparing_version = tensorflow_version.replace('tensorflow==', '')
        comparing_version = comparing_version.replace('tf-nightly==', '')
        comparing_version = comparing_version.replace('.dev', '-dev')
        is_installed = tensorflow.__version__ == comparing_version
    except: pass
    if (not is_installed):
        execute_script(['-m', 'pip', 'install', tensorflow_version])
    else:
        print(f'TensorFlow {tensorflow_version} is already installed')
    # Install pygit2
    is_installed = False
    try:
        import pygit2
        is_installed = pygit2.__version__ == '1.5.0'
    except: pass
    if (not is_installed):
        execute_script(['-m', 'pip', 'install', 'pygit2==1.5.0'])
        import pygit2
    else:
        print('pygit2 1.5.0 is already installed')
    # Directory of the TensorFlow object detection api and commit id
    od_api_dir = os.path.join(tempfile.gettempdir(), 'tensorflow-object-detection-api-2.4.1')
    od_api_ish = 'e356598a5b79a768942168b10d9c1acaa923bdb4'
    # Install the object detection api
    is_installed = False
    try:
        import object_detection
        repo = pygit2.Repository(od_api_dir)
        if (repo.head.target.hex == od_api_ish):
            is_installed = True
    except: pass
    # Install the TensorFlow models
    if (not is_installed):
        # Progress class for the git output
        class GitCallbacks(pygit2.RemoteCallbacks):
            def __init__(self, credentials=None, certificate=None):
                self.dateTime = datetime.datetime.now()
                return super().__init__(credentials=credentials, certificate=certificate)
            def transfer_progress(self, stats):
                now = datetime.datetime.now()
                if ((now - self.dateTime).total_seconds() > 1):
                    print('\rReceiving... Deltas [%d / %d], Objects [%d / %d]'%(stats.indexed_deltas, stats.total_deltas, stats.indexed_objects, stats.total_objects), end='', flush=True)
                    self.dateTime = now
                if (stats.received_objects >= stats.total_objects and stats.indexed_objects >= stats.total_objects and stats.indexed_deltas >= stats.total_deltas):
                    print('\r\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\rDone Deltas %d, Objects %d.'%(stats.total_objects, stats.total_objects))
                return super().transfer_progress(stats)
        # Create the callback for the progress
        callbacks = GitCallbacks();
        # Clone the TensorFlow models repository
        print('Cloning the TensorFlow object detection api repository')
        pygit2.clone_repository('https://github.com/tensorflow/models.git', od_api_dir, callbacks = callbacks)
        print('TensorFlow object detection api repository cloned')
        # Checkout the well known commit
        repo = pygit2.Repository(od_api_dir)
        (commit, reference) = repo.resolve_refish(od_api_ish)
        repo.checkout_tree(commit)
        repo.reset(pygit2.Oid(hex=od_api_ish), pygit2.GIT_RESET_HARD)
        # Move to the research dir
        currentDir = os.getcwd()
        os.chdir(os.path.join(od_api_dir, 'research'))
        # Install the protobuf tools
        execute_script(['-m', 'pip', 'install', 'grpcio-tools==1.32.0'])
        # Compile the protobufs
        import grpc_tools.protoc as protoc
        protoFiles = Path('object_detection/protos').rglob('*.proto')
        for protoFile in protoFiles:
            protoFilePath = str(protoFile)
            print('Compiling', protoFilePath)
            protoc.main(['grpc_tools.protoc', '--python_out=.', protoFilePath])
        # Install the object detection packages
        shutil.copy2('object_detection/packages/tf2/setup.py', '.')
        execute_script(['-m', 'pip', 'install', '.'])
        os.chdir(currentDir)
    else:
        print(f'TensorFlow object detection api SHA-1 {od_api_ish} is already installed')
    sys.path.append(os.path.join(od_api_dir, 'research'))
    sys.path.append(os.path.join(od_api_dir, 'research/slim'))
    sys.path.append(os.path.join(od_api_dir, 'research/object_detection'))
    print('Installation ok.')

if __name__ == '__main__':
    install_object_detection()

#@markdown ---
