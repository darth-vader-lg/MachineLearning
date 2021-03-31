# Module: install_virtual_environment.py
# Script for installing the Python virtual environment

# The name of the virtual environment
env_name = 'env'

def install_virtual_environment(env_name: str = env_name):
    """
    Install the virtual environment.
    Keyword arguments:
    env_name    -- the name of the virtual environment
    """
    import  os
    from    pathlib import Path
    import  subprocess
    import  sys
    # Creation of the virtual environment
    env_name = str(Path(env_name).absolute().resolve())
    if ('win32' in sys.platform):
        if (not os.path.isdir(env_name)):
            print('Creating the Python virtual environment')
            try:
                from utilities import execute_script
                execute_script(['-m', 'venv', env_name])
            except subprocess.CalledProcessError as exc:
                return exc.returncode
        # Adjust the environment paths
        if (os.path.dirname(sys.executable).lower() != os.path.join(env_name, 'Script').lower()):
            sys.executable = os.path.join(env_name, 'Scripts', os.path.basename(sys.executable))
        # Installation of the requirements
        from utilities import execute_script, install, get_package_info
        if (not get_package_info('tensorflow').name and os.path.isfile('requirements.txt')):
            print('Installing the requirements')
            try:
                execute_script(['-m', 'pip', 'install', '--upgrade', '-r', 'requirements.txt'])
            except subprocess.CalledProcessError as exc:
                return exc.returncode
    # Install the object detection environment
    if (not get_package_info('object-detection').version):
        print('Installing the object detection API')
        try:
            execute_script(['od_install.py'])
        except subprocess.CalledProcessError as exc:
            return exc.returncode
    return 0

if __name__ == '__main__':
    exit(install_virtual_environment(env_name))

