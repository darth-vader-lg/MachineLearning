# Module: install_virtual_environment.py
# Script for installing the Python virtual environment

import  os
from    pathlib import Path
import  subprocess
import  sys

# The name of the virtual environment
env_name = 'env'

def install_virtual_environment(env_name: str = env_name):
    """
    Install the virtual environment.
    Keyword arguments:
    env_name    -- the name of the virtual environment
    """
    # Creation of the virtual environment
    env_name = str(Path(env_name).absolute().resolve())
    if (not os.path.isdir(env_name)):
        print('Creating the Python virtual environment')
        try:
            from utilities import execute_script
            execute_script(['-m', 'venv', env_name])
        except subprocess.CalledProcessError as exc:
            return exc.returncode
    # Adjust the environment paths
    if (sys.exec_prefix.lower() != env_name.lower()):
        sys.exec_prefix = env_name
        sys.executable = os.path.join(env_name, 'Scripts', os.path.basename(sys.executable))
        sys.prefix = env_name
        script_dir = os.path.join(env_name, 'Scripts')
        #if (not script_dir.lower() in [str(Path(path).resolve()).lower() for path in sys.path]):
        sys.path.insert(0, os.path.join(env_name, 'Lib', 'site-packages'))
        sys.path.insert(0, script_dir)
        sys.path.insert(0, env_name)
        os.environ['PYTHONPATH'] = sys.path[0] + ';' + sys.path[1] + ';' + sys.path[2]
    # Install the pyinstaller
    from utilities import install, get_package_info
    pyinstaller = get_package_info('pyinstaller', env_name)
    if (not pyinstaller or not '4.2' in pyinstaller.versions):
        print('Installing PyInstaller')
        try:
            install('pyinstaller==4.2', env_name)
        except subprocess.CalledProcessError as exc:
            return exc.returncode
    # Install the object detection environment
    from utilities import get_package_info
    if (not get_package_info('object-detection', env_name)):
        print('Installing the object detection API')
        try:
            import od_install
            od_install.install_object_detection(env_name)
            #execute([sys.executable, 'od_install.py'])
        except subprocess.CalledProcessError as exc:
            return exc.returncode
    return 0

if __name__ == '__main__':
    exit(install_virtual_environment(env_name))

