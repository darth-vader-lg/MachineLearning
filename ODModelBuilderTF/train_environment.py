# Module train_environment.py
#@title #Train's environment initialization { form-width: "30%" }
#@markdown In this section the environment for the training will be initialized.
#@markdown
#@markdown All necessary directories will be crated and the Google drive
#@markdown containing the images will be mounted. Follow the instruction for the mounting during the execution.

import  os
from    pathlib import Path
import  sys

try:    from    default_cfg import Cfg
except: pass
try:    from    train_parameters import TrainParameters
except: pass

def init_train_environment(prm: TrainParameters):
    """
    Initialize the training environment with the right directories structure.
    Keyword arguments:
    prm     -- the train parameters
    """
    # Set the configuration for Google Colab
    if ('google.colab' in sys.modules and Cfg.data_on_drive):
        if (not os.path.exists('/mnt/MyDrive')):
            print('Mounting the GDrive')
            from google.colab import drive
            drive.mount('/mnt')
        # Check the existence of the train images dir
        gdrive_dir = os.path.join('/mnt', 'MyDrive', prm.train_images_dir)
        if (not os.path.isdir(gdrive_dir)):
            raise Exception('Error!!! The train images dir doesn`t exist')
        if (os.path.exists('/content/train-images')):
            os.unlink('/content/train-images')
        os.symlink(gdrive_dir, '/content/train-images', True)
        print(f"Google drive's {prm.train_images_dir} is linked to /content/train-images")
        prm.train_images_dir = '/content/train-images'
        # Check the existence of the evaluation images dir
        gdrive_dir = os.path.join('/mnt', 'MyDrive', prm.eval_images_dir)
        if (not os.path.isdir(gdrive_dir)):
            raise Exception('Error!!! The evaluation images dir doesn`t exist')
        if (os.path.exists('/content/eval-images')):
            os.unlink('/content/eval-images')
        os.symlink(gdrive_dir, '/content/eval-images', True)
        print(f"Google drive's {prm.eval_images_dir} is linked to /content/eval-images")
        prm.eval_images_dir = '/content/eval-images'
        # Check the existence of the output directory
        gdrive_dir = os.path.join('/mnt', 'MyDrive', prm.model_dir)
        if (not os.path.isdir(gdrive_dir)):
            print('Creating the output directory')
            os.mkdir(gdrive_dir)
        if (os.path.exists('/content/trained-model')):
            os.unlink('/content/trained-model')
        os.symlink(gdrive_dir, '/content/trained-model', True)
        print(f"Google drive's {prm.model_dir} is linked to /content/trained-model")
        prm.model_dir = '/content/trained-model'
    else:
        if (not os.path.isdir(prm.train_images_dir)):
            raise Exception('Error!!! The train images dir doesn`t exist')
        print(f'Train images from {str(Path(prm.train_images_dir).resolve())}')
        if (not os.path.isdir(prm.eval_images_dir)):
            raise Exception('Error!!! The evaluation images dir doesn`t exist')
        print(f'Train images from {str(Path(prm.eval_images_dir).resolve())}')
        if (not os.path.exists(prm.model_dir)):
            print('Creating the output directory')
            os.mkdir(prm.model_dir)
        print(f'The trained model will be in {str(Path(prm.model_dir).resolve())}')
    if (not os.path.exists(prm.annotations_dir)):
        os.mkdir(prm.annotations_dir)
    print(f'The annotations files will be in {str(Path(prm.annotations_dir).resolve())}')

if __name__ == '__main__':
    prm = ('prm' in locals() and isinstance(prm, TrainParameters) and prm) or TrainParameters.default
    init_train_environment(prm)

#@markdown ---
