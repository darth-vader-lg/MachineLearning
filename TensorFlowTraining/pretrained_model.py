# Module: pretrained_model.py

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
    if (not os.path.exists(prm.pre_trained_model_dir) or not os.path.exists(os.path.join(prm.pre_trained_model_dir, 'checkpoint', 'checkpoint'))):
        if (not os.path.exists(prm.pre_trained_model_base_dir)):
            os.mkdir(prm.pre_trained_model_base_dir)
        pre_trained_model_file = prm.pre_trained_model_dir + '.tar.gz'
        print(f'Downloading the pre-trained model {str(Path(pre_trained_model_file).name)}...')
        import tarfile
        request.urlretrieve(prm.model['DownloadPath'], pre_trained_model_file) # TODO: show progress
        print('Done.')
        print(f'Extracting the pre-trained model {str(Path(pre_trained_model_file).name)}...')
        tar = tarfile.open(pre_trained_model_file)
        tar.extractall(prm.pre_trained_model_base_dir)
        tar.close()
        os.remove(pre_trained_model_file)
    print(f'Pre-trained model is located at {str(Path(prm.pre_trained_model_dir).resolve())}')

if __name__ == '__main__':
    prm = ('prm' in locals() and prm) or BaseParameters.default
    download_pretrained_model(prm)
