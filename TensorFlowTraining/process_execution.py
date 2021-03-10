# Module process_execution.py

# Begin notebook cell
import subprocess
import sys

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

def execute_colab(fn):
    """
    Execute a function only in the Google Colab environment.
    Keyword arguments:
    fn      -- the function to execute
    """
    if ('google.colab' in sys.modules):
        fn()

def execute_non_colab(fn):
    """
    Execute a function only outside the Google Colab environment.
    Keyword arguments:
    fn      -- the function to execute
    """
    if (not 'google.colab' in sys.modules):
        fn()
# End notebook cell
