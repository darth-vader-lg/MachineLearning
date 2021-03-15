# Module: pretrained_model.py
#@title #Pre-trained model download { form-width: "20%" }
#@markdown Download of the pre-trained model from the TensorFlow 2 model zoo.

import  os
from    pathlib import Path
from    urllib import request

try:
    from    base_parameters import BaseParameters
except: pass

def download_pretrained_model(prm: BaseParameters):
    """
    Download from the TensorFlow model zoo the pre-trained model.
    Keyword arguments:
    prm     -- the base parameters
    """
    pre_trained_model_dir = os.path.join(prm.pre_trained_model_base_dir, prm.model['dir_name'])
    if (not os.path.exists(pre_trained_model_dir) or not os.path.exists(os.path.join(pre_trained_model_dir, 'checkpoint', 'checkpoint'))):
        if (not os.path.exists(prm.pre_trained_model_base_dir)):
            os.mkdir(prm.pre_trained_model_base_dir)
        pre_trained_model_file = pre_trained_model_dir + '.tar.gz'
        print(f'Downloading the pre-trained model {str(Path(pre_trained_model_file).name)}...')
        import tarfile
        request.urlretrieve(prm.model['download_path'], pre_trained_model_file) # TODO: show progress
        print('Done.')
        print(f'Extracting the pre-trained model {str(Path(pre_trained_model_file).name)}...')
        tar = tarfile.open(pre_trained_model_file)
        tar.extractall(prm.pre_trained_model_base_dir)
        tar.close()
        os.remove(pre_trained_model_file)
    print(f'Pre-trained model is located at {str(Path(pre_trained_model_dir).resolve())}')

if __name__ == '__main__':
    prm = ('prm' in locals() and prm) or BaseParameters.default
    download_pretrained_model(prm)

#@markdown ---
