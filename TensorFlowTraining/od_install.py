# Module od_install.py
#@title #Object detection libraries installation { form-width: "20%" }
#@markdown This step installs a well known Python environment for the train.

import  os
import  datetime
from    pathlib import Path
import  shutil
import  sys
import  tempfile

try:    from    default_cfg import Cfg
except: pass
try:    from    utilities import *
except: pass

def install_object_detection(env_path: str = None):
    """
    Install a well known environment.
    """
    if (env_path):
        env_path = str(Path(env_path).absolute().resolve())
    # Upgrade pip and setuptools
    is_installed = False
    try:
        is_installed = '21.0.1' in get_package_info('pip', env_path).versions
    except: pass
    if (not is_installed):
        install('pip==21.0.1', env_path)
    else:
        print('pip 21.0.1 is already installed')
    is_installed = False
    try:
        is_installed = '54.1.2' in get_package_info('setuptools', env_path).versions
    except: pass
    if (not is_installed):
        install('setuptools==54.1.2', env_path)
    else:
        print('setuptools 54.1.2 is already installed')
    # Install TensorFlow
    is_installed = False
    try:
        comparing_version = Cfg.tensorflow_version.replace('tensorflow==', '')
        comparing_version = comparing_version.replace('tf-nightly==', '')
        comparing_version = comparing_version.replace('.dev', '-dev')
        is_installed = comparing_version in get_package_info('tensorflow', env_path).versions
    except: pass
    if (not is_installed):
        install(Cfg.tensorflow_version, env_path)
    else:
        print(f'TensorFlow {Cfg.tensorflow_version} is already installed')
    # Install pygit2
    is_installed = False
    try:
        is_installed = '1.5.0' in get_package_info('pygit2', env_path).versions
    except: pass
    if (not is_installed):
        install('pygit2==1.5.0', env_path)
    else:
        print('pygit2 1.5.0 is already installed')
    import pygit2
    # Directory of the TensorFlow object detection api and commit id
    od_api_dir = os.path.join(tempfile.gettempdir(), 'tf-od-api-' + Cfg.od_api_git_sha1)
    # Install the object detection api
    is_installed = False
    try:
        if (get_package_info('object-detection', env_path)):
            repo = pygit2.Repository(od_api_dir)
            if (repo.head.target.hex == Cfg.od_api_git_sha1):
                is_installed = True
    except: pass
    # Install the TensorFlow models
    if (not is_installed):
        try:
            repo = pygit2.Repository(od_api_dir)
        except:
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
            repo = pygit2.Repository(od_api_dir)
        # Checkout the well known commit
        print(f'Checkout of the object detection api repository at the commit {Cfg.od_api_git_sha1}')
        (commit, reference) = repo.resolve_refish(Cfg.od_api_git_sha1)
        repo.checkout_tree(commit)
        repo.reset(pygit2.Oid(hex=Cfg.od_api_git_sha1), pygit2.GIT_RESET_HARD)
        # Move to the research dir
        currentDir = os.getcwd()
        os.chdir(os.path.join(od_api_dir, 'research'))
        # Install the protobuf tools
        install('grpcio-tools==1.32.0', env_path)
        # Compile the protobufs
        print(f'Compiling the protobufs')
        import grpc_tools.protoc as protoc
        protoFiles = Path('object_detection/protos').rglob('*.proto')
        for protoFile in protoFiles:
            protoFilePath = str(protoFile)
            print('Compiling', protoFilePath)
            protoc.main(['grpc_tools.protoc', '--python_out=.', protoFilePath])
        # Install the object detection packages
        print(f'Installing the object detection api.')
        shutil.copy2('object_detection/packages/tf2/setup.py', '.')
        install('.', env_path)
        os.chdir(currentDir)
    else:
        print(f'TensorFlow object detection api SHA-1 {Cfg.od_api_git_sha1} is already installed')
    # Append of the paths
    paths = [
        os.path.join(od_api_dir, 'research'),
        os.path.join(od_api_dir, 'research', 'slim'),
        os.path.join(od_api_dir, 'research', 'object_detection'),
        ]
    for path in paths:
        if (not path in sys.path):
            sys.path.append(path)
    print('Installation ok.')

if __name__ == '__main__':
    install_object_detection()

#@markdown ---
