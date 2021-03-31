# Module: build_exe.py
# Script to build the executable file for the object detection tarin and export

exe_name = 'ODModelBuilderTF'

def build_exe(name: str):
    try:
        # Start with an error code
        result = 1
        # Check if it's needed to build
        build = False
        import glob
        import os
        exe_file = os.path.join(exe_name, exe_name + '.exe')
        if (os.path.isfile(exe_file)):
            exe_file_time = os.path.getmtime(exe_file)
            for file in glob.glob('*.py'):
                if (os.path.getmtime(file) > exe_file_time):
                    build = True
                    break
        else:
            build = True
        # Create the executable
        if (build):
            # Ensure that the environment is correctly installed
            from install_virtual_environment import install_virtual_environment, env_name
            result = install_virtual_environment()
            if (result != 0):
                return result
            result = 1
            from utilities import execute
            execute([
                'pyinstaller',
                '--onefile',
                '--hidden-import', 'pandas._libs.tslibs.base',
                '--hidden-import', 'tensorflow.python.keras.engine.base_layer_v1',
                '--add-binary', f'{env_name}/Scripts/tensorboard.exe;.',
                '--runtime-tmpdir', '%TEMP%/tf-od-model-builder',
                '--distpath', exe_name,
                '--name', exe_name,
                'main.py'])
            # Remove the build directory
            import shutil
            shutil.rmtree('build')
        # Ok
        result = 0
    except Exception as exc:
        print(exc)
    return result

if __name__ == '__main__':
    exit(build_exe(exe_name))


