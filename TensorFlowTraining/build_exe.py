# Module: build_exe.py
# Script to build the executable file for the object detection tarin and export

from    install_virtual_environment import *

exe_name = 'tf-od-model-builder'

def build_exe(name: str):
    result = install_virtual_environment()
    if (result != 0):
        return result
    result = 1
    try:
        import PyInstaller.__main__
        result = PyInstaller.__main__.run([
            '--onefile',
            '--clean',
            '--hidden-import', 'pandas._libs.tslibs.base',
            '--hidden-import', 'tensorflow.python.keras.engine.base_layer_v1',
            '--add-binary', f'{env_name}/Scripts/tensorboard.exe;.',
            '--runtime-tmpdir', '%TEMP%/tf-od-model-builder',
            '--distpath', exe_name,
            '--name', exe_name,
            'main.py'
        ])
        result = 0
    except Exception as exc:
        print(exc)
    return result

if __name__ == '__main__':
    exit(build_exe(exe_name))


