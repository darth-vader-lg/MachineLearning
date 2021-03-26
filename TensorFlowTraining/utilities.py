# Module: utilities.py
#@title #Utility functions
#@markdown Some utility functions used for the train steps.

import  os
from    pathlib import Path
import  subprocess
import  sys

def allow_flags_override():
    """
    Allow the argv flags override.
    """
    # Avoiding the absl error for duplicated flags if defined more than one time
    from absl import flags
    for f in flags.FLAGS.flag_values_dict():
        flags.FLAGS[f].allow_override = True

def execute_subprocess(cmd):
    """
    Execute a subprocess returning each line of the standard output.
    Keyword arguments:
    cmd     -- the process to execute with its parameters
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def execute(cmd):
    """
    Execute a subprocess printing its standard output.
    Keyword arguments:
    cmd     -- the process to execute with its parameters
    """
    for output in execute_subprocess(cmd):
        print(output, end="")

def execute_script(cmd):
    """
    Execute a script as a subprocess printing its standard output.
    Keyword arguments:
    cmd     -- the parameters of the script
    """
    script_cmd = ['python']
    script_cmd.extend(cmd)
    for output in execute_subprocess(script_cmd):
        print(output, end="")

def get_package_info(package_name: str, env_path: str = None):
    """
    Return a package and its version if installed. Otherwise None
    """
    packages_dir = os.path.join(env_path or sys.prefix, 'Lib', 'site-packages')
    import pkg_resources
    dists = [dist for dist in pkg_resources.find_distributions(packages_dir) if dist.key == package_name]
    if (len(dists) > 0):
        class Result(object):
            def __init__(self, *args, **kwargs):
               self.name = dists[0].key
               self.versions = [dist.version for dist in dists]
               return super().__init__(*args, **kwargs)
        return Result()
    return None

def get_type_of_script():
    """
    Return of the type of the script is being executed
    """
    try:
        ipy_str = str(type(get_ipython()))
        if ('ipykernel_launcher.py' in sys.argv[0]):
            return 'jupyter'
        return 'ipython'
    except:
        if ('python' in Path(sys.executable).name.lower()):
            return 'terminal'
        else:
            return "executable"

def install(package: str, env_path: str = None):
    """
    Launch the installer process
    """
    cmd = ['-m', 'pip', 'install']
    if (env_path):
        env_path = str(Path(env_path).absolute().resolve())
        site_packages = os.path.join(env_path, 'Lib', 'site-packages')
        #@@@cmd.extend(['--target', site_packages])
        cmd.insert(0, os.path.join(env_path, 'Scripts', os.path.basename(sys.executable)))
    else:
        cmd.insert(0, 'python')
    cmd.extend(['--upgrade', package])
    execute(cmd)
    return

def is_executable():
    """
    True if running as an executable
    """
    return get_type_of_script() == 'executable'

def is_ipython():
    """
    True if running in an ipython environment
    """
    return get_type_of_script() == 'ipython'

def is_jupyter():
    """
    True if running in a jupyter notebook
    """
    return get_type_of_script() == 'jupyter'

def is_terminal():
    """
    True if running a terminal environment
    """
    return get_type_of_script() == 'terminal'

if __name__ == '__main__':
    print('Utilities functions initialized')

#@markdown ---
