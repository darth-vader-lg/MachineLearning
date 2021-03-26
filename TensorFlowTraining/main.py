# Module: main.py
# Main script for training and exporting a model

from    pathlib import Path

try:    from    default_cfg import Cfg
except: pass
try:    from    utilities import *
except: pass

def main(unused_argv):
    # Clear the default configuration output dirs.
    # They must be specified in the command line parameters for enabling phases.
    if (is_executable()):
        Cfg.exported_model = None
        Cfg.max_train_steps = 0
    try:
        if (not is_executable()):
            print('=' * 80)
            print('Train')
        from    train_main import train_main
        train_main(unused_argv)
    except (KeyboardInterrupt, SystemExit):
        pass
    try:
        if (not is_executable()):
            print('=' * 80)
            print('=' * 80)
            print('=' * 80)
            print('=' * 80)
            print('Export to pb SavedModel')
        from    export_main import export_main
        export_main(unused_argv)
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == '__main__':
    if (not is_jupyter() and not is_executable()):
        from od_install import install_object_detection
        install_object_detection()
    try:
        import tensorflow as tf
        # import the module here just for having the flags defined
        if (not is_jupyter()):
            allow_flags_override()
            from object_detection import model_main_tf2
            allow_flags_override()
            from object_detection import exporter_main_v2
            # Function that restore the original mandatory flags values and call the export main
            from absl import flags
            fake_flags = ['pipeline_config_path', 'trained_checkpoint_dir', 'output_directory']
            def restore_and_run(unused_argv):
                for flag in fake_flags:
                    flags.FLAGS[flag].value = flags.FLAGS[flag].value if flags.FLAGS[flag].value != '?!?' else None
                    flags.FLAGS[flag].validators.clear()
                main(unused_argv)
            # Put fake values fot the mandatory flags and run the restore and run function
            for flag in fake_flags:
                flags.FLAGS[flag].value = '?!?'
            tf.compat.v1.app.run(restore_and_run)
        else:
            tf.compat.v1.app.run(main)
    except KeyboardInterrupt:
        if (not is_executable()):
            print('Interrupted by user')
    except SystemExit:
        if (not is_executable()):
            print('End')
