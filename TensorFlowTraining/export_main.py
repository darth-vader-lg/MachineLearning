# Module: export_main.py
#@title #Export { form-width: "20%" }
#@markdown The export action. It export the last checkpoint in a saved model.

from    absl import flags
import  os
import  sys

try:    from    utilities import *
except: pass

# Avoiding the absl error for duplicated flags if run again the cell from a notebook
allow_flags_override()

def export_main(unused_argv):
    # Part of code not executed on Colab notebook
    def run_py_mode():
        # Init the train environment
        from export_environment import init_export_environment
        from export_parameters import ExportParameters
        export_parameters = ExportParameters()
        export_parameters.update_values()
        # Check if the export directory is specified
        if (not export_parameters.output_directory or len(export_parameters.output_directory) < 1):
            return
        init_export_environment(export_parameters)
        # Import the export main function
        from object_detection import exporter_main_v2
        export_parameters.update_flags()
        # Export the model
        exporter_main_v2.main(unused_argv)
    def run_notebook_mode():
        # Check if the export directory is specified
        if (not export_parameters.output_directory or len(export_parameters.output_directory) < 1):
            return
        # Import the export main function
        from object_detection import exporter_main_v2
        prm.update_flags()
        # Export the model
        exporter_main_v2.main(unused_argv)
    # Execution
    if (is_jupyter()):
        run_notebook_mode()
    else:
        run_py_mode()

if __name__ == '__main__':
    if (not is_jupyter() and not is_executable()):
        from od_install import install_object_detection
        install_object_detection()
    try:
        # Import of the TensorFlow module
        import tensorflow as tf
        # import the module here just for having the flags defined
        if (not is_jupyter()):
            # Allow the ovverride and save the current values of the mandatory flags
            allow_flags_override()
            # Import the module for defining the flags
            from object_detection import exporter_main_v2
            # Function that restore the original mandatory flags values and call the export main
            fake_flags = ['pipeline_config_path', 'trained_checkpoint_dir', 'output_directory']
            def restore_and_run(unused_argv):
                for flag in fake_flags:
                    flags.FLAGS[flag].value = flags.FLAGS[flag].value if flags.FLAGS[flag].value != '?!?' else None
                export_main(unused_argv)
            # Put fake values fot the mandatory flags and run the restore and run function
            for flag in fake_flags:
                flags.FLAGS[flag].value = '?!?'
            tf.compat.v1.app.run(restore_and_run)
        else:
            # Run the export main
            tf.compat.v1.app.run(export_main)
    except KeyboardInterrupt:
        if (not is_executable()):
            print('Export interrupted by user')
    except SystemExit:
        if (not is_executable()):
            print('Export complete')
    else:
        if (not is_executable()):
            print('Export complete')

#@markdown ---
