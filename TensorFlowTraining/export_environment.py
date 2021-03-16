# Module export_environment.py
#@title #Export's environment initialization { form-width: "30%" }
#@markdown In this section the environment for the export will be initialized.
#@markdown
#@markdown All necessary directories will be mounted from the Google drive.
#@markdown Follow the instruction for the mounting during the execution.

import  os
from    pathlib import Path
import  shutil
import  sys

try:    from    default_cfg import *
except: pass
try:    from    export_parameters import ExportParameters
except: pass

def init_export_environment(prm: ExportParameters):
    """
    Initialize the model export environment with the right directories structure.
    Keyword arguments:
    prm     -- the export parameters
    """
    # Set the configuration for Google Colab
    if ('google.colab' in sys.modules and cfg_data_on_drive):
        if (not os.path.exists('/mnt/MyDrive')):
            print('Mounting the GDrive')
            from google.colab import drive
            drive.mount('/mnt')

        # Check the existence of the checkpoints directory
        gdrive_dir = os.path.join('/mnt', 'MyDrive', prm.trained_checkpoint_dir)
        if (not os.path.isdir(gdrive_dir)):
            raise Exception('Error!!! The trained checkpoint dir doesn`t exist')
        if (os.path.exists('/content/trained-model')):
            os.unlink('/content/trained-model')
        os.symlink(gdrive_dir, '/content/trained-model', True)
        print(f"Google drive's {prm.trained_checkpoint_dir} is linked to /content/trained-model")
        prm.trained_checkpoint_dir = '/content/trained-model'
        # Check the existence of the output directory
        gdrive_dir = os.path.join('/mnt', 'MyDrive', prm.output_directory)
        if (not os.path.isdir(gdrive_dir)):
            print('Creating the output directory')
            os.mkdir(gdrive_dir)
        if (str(Path(prm.output_directory).resolve()) == str(Path(prm.model_dir).resolve())):
            raise Exception("Error: export directory cannot be the train directory")
        if (os.path.exists('/content/exported-model')):
            os.unlink('/content/exported-model')
        os.symlink(gdrive_dir, '/content/exported-model', True)
        gdrive_dir = os.path.join(prm.output_directory, 'exported-model')
        print(f"Google drive's {gdrive_dir} is linked to /content/exported-model")
        prm.output_directory = '/content/exported-model'
    else:
        if (not os.path.isdir(prm.trained_checkpoint_dir)):
            raise Exception('Error!!! The trained checkpoint dir doesn`t exist')
        print(f'Trained checkpoint directory from {str(Path(prm.trained_checkpoint_dir).resolve())}')
        if (not os.path.exists(prm.output_directory)):
            print('Creating the output directory')
            os.mkdir(prm.output_directory)
        if (str(Path(prm.output_directory).resolve()) == str(Path(prm.model_dir).resolve())):
            raise Exception("Error: export directory cannot be the train directory")
        print(f'The trained model will be in {str(Path(prm.model_dir).resolve())}')
    # Copy the label file in the export directory
    shutil.copy2(os.path.join(prm.trained_checkpoint_dir, 'label_map.pbtxt'), prm.output_directory)

if __name__ == '__main__':
    prm = ('prm' in locals() and isinstance(prm, ExportParameters) and prm) or ExportParameters.default
    init_export_environment(prm)

#@markdown ---
