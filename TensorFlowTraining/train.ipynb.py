# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ySpmFQsm4wKUb92Bbo1TbTl7zxxjEBBf

---

# Object detection train with TensorFlow 2.4.1
This notebook will train a model for the object detection purpouse.
It can be run as a Jupyter notebook in the Google Colab environment or exported as a Python file and run from a command line.

This software detects automatically if you are working on a Colab environment or in your local machine.

For a local machine it just requires a Python >= 3.7 installed.

All the operations for installing all the required libraries and for preparing the data needed by the train algoritm will be done effortlessly for you.
## Train preparation:
*   Collect a set of images containing the objects that you want to train.
*   Split the set in two different folders; one for the train and the other for the evaluation. The number of the evaluation images could be from 10% to 30% of the train images.
*   Label the images using a standard images annotation tool as [labelImg](https://github.com/tzutalin/labelImg), [tVoTT](https://github.com/microsoft/VoTT), etc... and save the xml for each picture in the Coco format. 
*   Copy the folders with the prepared images set in your GDrive (if you are working on a Colab environment).
*   Configure the train parameters listed in the next notebook's cell.

## Train:
Run the process and enjoy your time waiting for the train will be complete.

You can also stop the train and restart again later; if you didn't clean the output directory for the model the train will restart from the last checkpoint, continuing the fine tuning of the model.
The progress of the train can be followed by the Tensorboard (already included in this notebook).

### For Colab environment train:
The notebook needs to mount your GDrive. It will ask you the access authorization. Follow the instructions.

---
"""

#@title #Notebook configuration
#@markdown ## Base model:
#@markdown (The base model from which the train will start)
model = 'SSD MobileNet v2 320x320' #@param ['SSD MobileNet v2 320x320', 'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)']
#@markdown ---
#@markdown ## Target directory:
#@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where the checkpoints will be saved.
trained_model = 'trained-model' #@param {type:"string"}
#@markdown ---
#@markdown ## Images directories:
#@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where is located the images set for the train and for the evaluation.
train_images_dir = 'images/train' #@param {type:"string"}
eval_images_dir = 'images/eval' #@param {type:"string"}

import  os
import  sys
if ('google.colab' in sys.modules):
    if (not os.path.exists('/mnt/MyDrive')):
        print('Mounting the GDrive')
        from google.colab import drive
        drive.mount('/mnt')
#@markdown ---

# Module: model_types.py
#@title #Model types { vertical-output: true, form-width: "20%" }
#@markdown Initialize the list of the available pre-trained models and their parameters.

""" List of the available models and their definitions """
models = {
    'SSD MobileNet v2 320x320': {
        'DownloadPath': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 12,
        'height': 300,
        'width': 300
    },
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)': {
        'DownloadPath': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
}

if __name__ == '__main__':
    import pprint
    pprint.PrettyPrinter(1).pprint(models)
    print('Dictionary of pre-trained models configured')

#@markdown ---

# Module: utilities.py
#@title #Utility functions
#@markdown Some utility functions used for the train steps.

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

if __name__ == '__main__':
    print('Utilities functions initialized')

#@markdown ---

# Module: base_parameters.py
#@title #Base parameters { form-width: "20%" }
#@markdown Definition of the base parameters class.

from    absl import flags
import  os
from    pathlib import Path
import  sys
import  tempfile

try:
    from    model_types import models
except: pass

class BaseParameters:
    """ Class holding the base parameters """
    def __init__(self):
        """ Constructor """
        self._model_type = ('model' in globals() and model) or 'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)'
        self._model_dir = ('trained_model' in globals() and trained_model) or 'trained-model'
        self._train_images_dir = ('train_images_dir' in globals() and train_images_dir) or 'images/train'
        self._eval_images_dir = ('eval_images_dir' in globals() and eval_images_dir) or 'images/eval'
        self._annotations_dir = 'annotations'
        self._pre_trained_model_base_dir = os.path.join(tempfile.gettempdir(), 'tensorflow-pre-trained-models')
        self._is_path = [
            'model_dir',
            'train_images_dir',
            'eval_images_dir',
            'annotations_dir']
    default = None
    @property
    def model(self):
        global models
        return models[self.model_type]
    @property
    def model_type(self): return self._model_type
    @model_type.setter
    def model_type(self, value): self._model_type = value
    @property
    def model_dir(self): return self._model_dir
    @model_dir.setter
    def model_dir(self, value): self._model_dir = value
    @property
    def train_images_dir(self): return self._train_images_dir
    @train_images_dir.setter
    def train_images_dir(self, value): self._train_images_dir = value
    @property
    def eval_images_dir(self): return self._eval_images_dir
    @eval_images_dir.setter
    def eval_images_dir(self, value): self._eval_images_dir = value
    @property
    def annotations_dir(self): return self._annotations_dir
    @annotations_dir.setter
    def annotations_dir(self, value): self._annotations_dir = value
    @property
    def pre_trained_model_base_dir(self): return self._pre_trained_model_base_dir
    @pre_trained_model_base_dir.setter
    def pre_trained_model_base_dir(self, value): self._pre_trained_model_base_dir = value
    @property
    def pre_trained_model_dir(self):
        return str(Path(os.path.join(
            self.pre_trained_model_base_dir,
            Path(self.model["DownloadPath"]).name)).with_suffix("").with_suffix(""))
    def __str__(self):
        result = ''
        propnames = [p for p in dir(type(self)) if isinstance(getattr(type(self), p),property) and getattr(self, p)]
        for prop in propnames:
            try:
                value = getattr(self, prop)
                if (prop in self._is_path):
                    value = str(Path(value).resolve())
                if (len(result) > 0):
                    result += '\n'
                result += f'{prop}: {value}'
            except:
                pass
        return result
    def update_flags(self):
        propnames = [p for p in dir(type(self)) if isinstance(getattr(type(self), p),property) and getattr(self, p)]
        for prop in propnames:
            try:
                value = getattr(self, prop)
                if (value):
                    setattr(flags.FLAGS, prop, value)
                    print(f'Written flag {prop} with value {value}')
            except:
                pass
    def update_values(self):
        propnames = [p for p in dir(type(self)) if isinstance(getattr(type(self), p),property)]
        for prop in propnames:
            try:
                value = getattr(flags.FLAGS, prop)
                if (value):
                    setattr(self, prop, value)
                    print(f'Written property {prop} with value {value}')
            except:
                pass
BaseParameters.default = BaseParameters.default or BaseParameters()

if __name__ == '__main__':
    prm = ('prm' in locals() and prm and isinstance(prm, BaseParameters)) or BaseParameters.default
    print(prm)
    print('Base parameters configured')

#@markdown ---

#module train_parameters.py
#@title #Train parameters { form-width: "20%" }
#@markdown Definition of the train parameters. Read the comments in the flags
#@markdown section of the train main module
#@markdown https://raw.githubusercontent.com/tensorflow/models/e356598a5b79a768942168b10d9c1acaa923bdb4/research/object_detection/model_main_tf2.py

import  os

try:
    from    base_parameters import BaseParameters
except: pass

class TrainParameters(BaseParameters):
    """ Class holding the train execution parameters """
    def __init__(self):
        """ Constructor """
        super().__init__()
        self._pipeline_config_path = os.path.join(self.annotations_dir, 'pipeline.config')
        self._num_train_steps = None
        self._eval_on_train_data = False
        self._sample_1_of_n_eval_examples = None
        self._sample_1_of_n_eval_on_train_examples = 5
        self._checkpoint_dir = None
        self._eval_timeout = 3600
        self._use_tpu = False
        self._tpu_name = None
        self._num_workers = 1
        self._checkpoint_every_n = 1000
        self._record_summaries = True
        self._is_path.extend([
            'pipeline_config_path',
            'checkpoint_dir'])
    default = None
    @property
    def pipeline_config_path(self): return self._pipeline_config_path
    @pipeline_config_path.setter
    def pipeline_config_path(self, value): self._pipeline_config_path = value
    @property
    def num_train_steps(self): return self._num_train_steps
    @num_train_steps.setter
    def num_train_steps(self, value): self._num_train_steps = value
    @property
    def eval_on_train_data(self): return self._eval_on_train_data
    @eval_on_train_data.setter
    def eval_on_train_data(self, value): self._eval_on_train_data = value
    @property
    def sample_1_of_n_eval_examples(self): return self._sample_1_of_n_eval_examples
    @sample_1_of_n_eval_examples.setter
    def sample_1_of_n_eval_examples(self, value): self._sample_1_of_n_eval_examples = value
    @property
    def sample_1_of_n_eval_on_train_examples(self): return self._sample_1_of_n_eval_on_train_examples
    @sample_1_of_n_eval_on_train_examples.setter
    def sample_1_of_n_eval_on_train_examples(self, value): self._sample_1_of_n_eval_on_train_examples = value
    @property
    def checkpoint_dir(self): return self._checkpoint_dir
    @checkpoint_dir.setter
    def checkpoint_dir(self, value): self._checkpoint_dir = value
    @property
    def eval_timeout(self): return self._eval_timeout
    @eval_timeout.setter
    def eval_timeout(self, value): self._eval_timeout = value
    @property
    def use_tpu(self): return self._use_tpu
    @use_tpu.setter
    def use_tpu(self, value): self._use_tpu = value
    @property
    def tpu_name(self): return self._tpu_name
    @tpu_name.setter
    def tpu_name(self, value): self._tpu_name = value
    @property
    def num_workers(self): return self._num_workers
    @num_workers.setter
    def num_workers(self, value): self._num_workers = value
    @property
    def checkpoint_every_n(self): return self._checkpoint_every_n
    @checkpoint_every_n.setter
    def checkpoint_every_n(self, value): self._checkpoint_every_n = value
    @property
    def record_summaries(self): return self._record_summaries
    @record_summaries.setter
    def record_summaries(self, value): self._record_summaries = value
TrainParameters.default = TrainParameters.default or TrainParameters()

if __name__ == '__main__':
    prm = ('prm' in locals() and prm and isinstance(prm, TrainParameters)) or TrainParameters.default
    print(prm)
    print('Train parameters configured')

#@markdown ---

# Module od_install.py
#@title #Object detection libraries installation { form-width: "20%" }
#@markdown This step installs a well known Python environment for the train.

import os
import datetime
from   pathlib import Path
import shutil
import sys
import tempfile

try:
    from    utilities import execute
except: pass

def install_object_detection():
    """
    Install a well known environment.
    """
    # Path of the python interpreter executable
    pythonPath = os.path.join(os.path.dirname(sys.executable), 'python3')
    if (not os.path.exists(pythonPath)):
        pythonPath = os.path.join(os.path.dirname(sys.executable), 'python')
    # Upgrade pip and setuptools
    execute([pythonPath, '-m', 'pip', 'install', '--upgrade', 'pip==21.0.1'])
    execute([pythonPath, '-m', 'pip', 'install', '--upgrade', 'setuptools==54.0.0'])
    # Install TensorFlow
    execute([pythonPath, '-m', 'pip', 'install', 'tensorflow==2.4.1'])
    # Install pygit2
    execute([pythonPath, '-m', 'pip', 'install', 'pygit2==1.5.0'])
    import pygit2
    # Progress class for the git output
    class GitCallbacks(pygit2.RemoteCallbacks):
        def __init__(self, credentials=None, certificate=None):
            self.dateTime = datetime.datetime.now()
            return super().__init__(credentials=credentials, certificate=certificate)
        def transfer_progress(self, stats):
            now = datetime.datetime.now()
            if ((now - self.dateTime).total_seconds() > 1):
                print('\rReceiving... Deltas [%d / %d], Objects [%d / %d]'%(stats.indexed_deltas, stats.total_deltas, stats.indexed_objects, stats.total_objects), end='', flush=True)
                self.dateTime = now
            if (stats.received_objects >= stats.total_objects and stats.indexed_objects >= stats.total_objects and stats.indexed_deltas >= stats.total_deltas):
                print('\r\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\rDone Deltas %d, Objects %d.'%(stats.total_objects, stats.total_objects))
            return super().transfer_progress(stats)
    # Directory of the TensorFlow object detection api
    odApiDir = os.path.join(tempfile.gettempdir(), 'tensorflow-object-detection-api-2.4.1')
    # Install the TensorFlow models
    if (not os.path.isdir(odApiDir)):
        # Create the callback for the progress
        callbacks = GitCallbacks();
        # Clone the TensorFlow models repository
        print('Cloning the TensorFlow models repository')
        pygit2.clone_repository('https://github.com/tensorflow/models.git', odApiDir, callbacks = callbacks)
        print('TensorFlow models repository cloned')
        # Checkout a well known commit
        repo = pygit2.Repository(odApiDir)
        ish = 'e356598a5b79a768942168b10d9c1acaa923bdb4'
        (commit, reference) = repo.resolve_refish(ish)
        repo.checkout_tree(commit)
        repo.reset(pygit2.Oid(hex=ish), pygit2.GIT_RESET_HARD)
        # Move to the research dir
        currentDir = os.getcwd()
        os.chdir(os.path.join(odApiDir, 'research'))
        # Install the protobuf tools
        execute([pythonPath, '-m', 'pip', 'install', 'grpcio-tools==1.32.0'])
        # Compile the protobufs
        import grpc_tools.protoc as protoc
        protoFiles = Path('object_detection/protos').rglob('*.proto')
        for protoFile in protoFiles:
            protoFilePath = str(protoFile)
            print('Compiling', protoFilePath)
            protoc.main(['grpc_tools.protoc', '--python_out=.', protoFilePath])
        # Install the object detection packages
        shutil.copy2('object_detection/packages/tf2/setup.py', '.')
        execute([pythonPath, '-m', 'pip', 'install', '.'])
        os.chdir(currentDir)
    sys.path.append(os.path.join(odApiDir, 'research'))
    sys.path.append(os.path.join(odApiDir, 'research/slim'))
    sys.path.append(os.path.join(odApiDir, 'research/object_detection'))
    print('Installation completed.')

if __name__ == '__main__':
    install_object_detection()

#@markdown ---

# Module train_environment.py
#@title #Environment initialization { form-width: "30%" }
#@markdown In this section the environment for the training will be initialized.
#@markdown
#@markdown All necessary directories will be crated and the Google drive
#@markdown containing the images will be mounted. Follow the instruction for the mounting during the execution.

import  os
from    pathlib import Path
import  sys

try:
    from    train_parameters import TrainParameters
except: pass

def init_train_environment(prm: TrainParameters):
    """
    Initialize the training environment with the right directories structure.
    Keyword arguments:
    prm     -- the train parameters
    """
    # Set the configuration for Google Colab
    if ('google.colab' in sys.modules):
        if (not os.path.exists('/mnt/MyDrive')):
            print('Mounting the GDrive')
            from google.colab import drive
            drive.mount('/mnt')
        # Check the existence of the train images dir
        gdriveOutputDir = os.path.join('/mnt', 'MyDrive', prm.train_images_dir)
        if (not os.path.isdir(gdriveOutputDir)):
            raise Exception('Error!!! The train images dir doesn`t exist')
        if (os.path.exists('/content/train-images')):
            os.unlink('/content/train-images')
        os.symlink(gdriveOutputDir, '/content/train-images', True)
        print(f"Google drive's {prm.train_images_dir} is linked to /content/train-images")
        prm.train_images_dir = '/content/train-images'
        # Check the existence of the evaluation images dir
        gdriveOutputDir = os.path.join('/mnt', 'MyDrive', prm.eval_images_dir)
        if (not os.path.isdir(gdriveOutputDir)):
            raise Exception('Error!!! The evaluation images dir doesn`t exist')
        if (os.path.exists('/content/eval-images')):
            os.unlink('/content/eval-images')
        os.symlink(gdriveOutputDir, '/content/eval-images', True)
        print(f"Google drive's {prm.eval_images_dir} is linked to /content/eval-images")
        prm.eval_images_dir = '/content/eval-images'
        # Check the existence of the output directory
        gdriveOutputDir = os.path.join('/mnt', 'MyDrive', prm.model_dir)
        if (not os.path.isdir(gdriveOutputDir)):
            print('Creating the output directory')
            os.mkdir(gdriveOutputDir)
        if (os.path.exists('/content/trained-model')):
            os.unlink('/content/trained-model')
        os.symlink(gdriveOutputDir, '/content/trained-model', True)
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
            print('Creating the output dir')
            os.mkdir(prm.model_dir)
        print(f'The trained model will be in {str(Path(prm.model_dir).resolve())}')
    if (not os.path.exists(prm.annotations_dir)):
        os.mkdir(prm.annotations_dir)
    print(f'The annotations files will be in {str(Path(prm.annotations_dir).resolve())}')

if __name__ == '__main__':
    prm = ('prm' in locals() and prm) or TrainParameters.default
    init_train_environment(prm)

#@markdown ---

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

#@markdown ---

# Module: tf_record.py
#@title #TensorFlow's records { form-width: "30%" }
#@markdown In this step there will be created the TensorFlow records from the
#@markdown annotated images and the file contained all the labels' indices.

import  glob
import  io
import  os
from    pathlib import Path
import  shutil

try:
    from    base_parameters import BaseParameters
except: pass

class TFRecord:
    """Class for the TensorFlow records creation"""
    def __init__(self):
        """ Constructor """
        super().__init__()
        self._label_set = set()
        self._label_dict = dict()
    def class_text_to_int(self, row_label):
        """
        Convertion of the text of the labels to an integer index
        Keyword arguments:
        row_label   -- the label to convert to int
        """
        if (len(self._label_dict) == 0):
            count = len(self._label_set)
            labelIx = 1
            for label in self._label_set:
                self._label_dict[label] = labelIx
                labelIx += 1
        return self._label_dict[row_label]
    def create_tf_example(self, group, path):
        """
        TensorFlow example creator
        Keyword arguments:
        group   -- group's name
        path    -- path of the labeled images
        """
        from object_detection.utils import dataset_util
        from PIL import Image
        import tensorflow as tf
        with tf.compat.v1.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.class_text_to_int(row['class']))
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example
    def create_tf_record(self, image_dir, output_file, labels_file = None, csv_file = None):
        """
        TensorFlow record creator
        Keyword arguments:
        image_dir   -- the directory containing the images
        output_file -- the output file path and name
        labels_file -- the optional output file path and name of the resulting labels file
        csv_file    -- the optional output file path and name of the csv file
        """
        import tensorflow as tf
        writer = tf.compat.v1.python_io.TFRecordWriter(output_file)
        path = os.path.join(image_dir)
        examples = self.xml_to_csv(image_dir)
        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print(f'Created the TFRecord file {str(Path(output_file).resolve())}')
        if labels_file is not None:
            from google.protobuf import text_format
            from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
            msg = StringIntLabelMap()
            for id, name in enumerate(self._label_set, start = 1):
                msg.item.append(StringIntLabelMapItem(id = id, name = name))
            text = str(text_format.MessageToBytes(msg, as_utf8 = True), 'utf-8')
            with open(labels_file, 'w') as f:
                f.write(text)
            print(f'Created the labels map file {str(Path(labels_file).resolve())}')
        if csv_file is not None:
            examples.to_csv(csv_file, index = None)
            print(f'Created the CSV file {str(Path(csv_file).resolve())}')
    def split(self, df, group):
        """
        Split the labels in an image
        Keyword arguments:
        df      -- TensorFlow example
        group   -- group's name
        """
        from collections import namedtuple
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
    def xml_to_csv(self, path):
        """
        Convert the xml files generated by labeling image softwares into the cvs panda format
        Keyword arguments:
        path    -- Path of the generated csv file
        """
        import pandas as pd
        import xml.etree.ElementTree as ET
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (
                    root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text))
                xml_list.append(value)
                self._label_set.add(member[0].text)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns = column_name)
        return xml_df

def create_tf_records(prm: BaseParameters):
    """
    TensorFlow record files creator
    Keyword arguments:
    prm     -- Parameters
    """
    print("Creating TFRecord for the train images...")
    TFRecord().create_tf_record(
        prm.train_images_dir,
        os.path.join(prm.annotations_dir, 'train.record'),
        os.path.join(prm.annotations_dir, 'label_map.pbtxt'))
    print("Creating TFRecord for the evaluation images...")
    TFRecord().create_tf_record(
        prm.eval_images_dir,
        os.path.join(prm.annotations_dir, 'eval.record'))
    shutil.copy2(os.path.join(prm.annotations_dir, 'label_map.pbtxt'), prm.model_dir)
    print(f"The labels map file was copied to {(os.path.join(str(Path(prm.model_dir).resolve()), 'label_map.pbtxt'))}")

if __name__ == '__main__':
    prm = ('prm' in locals() and prm) or BaseParameters.default
    create_tf_records(prm)

#@markdown ---

# Module: train_pipeline
#@title #Train pipeline configuration { form-width: "20%" }
#@markdown Configuration of the train pipeline using the original train pipeline
#@markdown of the pre-trained model but modifing some parameters as paths,
#@markdown number of labels, etc... 

import  os
import  shutil

try:
    from    train_parameters import TrainParameters
except: pass

def config_train_pipeline(prm: TrainParameters):
    """
    Configure the training pipeline
    Keyword arguments:
    prm     -- Parameters
    """
    import tensorflow as tf
    from object_detection.protos import pipeline_pb2
    from object_detection.utils import label_map_util
    from google.protobuf import text_format
    # Copy the pipeline configuration file if it's not already present in the output dir
    print('Configuring the pipeline')
    output_file = prm.pipeline_config_path
    shutil.copy2(os.path.join(prm.pre_trained_model_dir, 'pipeline.config'), output_file)
    # Read the number of labels
    label_dict = label_map_util.get_label_map_dict(os.path.join(prm.annotations_dir, 'label_map.pbtxt'))
    labels_count = len(label_dict)
    # Configuring the pipeline
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(output_file, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    pipeline_config.model.ssd.num_classes = labels_count
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = prm.model['height']
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = prm.model['width']
    pipeline_config.train_config.batch_size = prm.model['batch_size']
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(prm.pre_trained_model_dir, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path = os.path.join(prm.annotations_dir, 'label_map.pbtxt')
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = os.path.join(prm.annotations_dir, 'train.record')
    pipeline_config.eval_input_reader[0].label_map_path = os.path.join(prm.annotations_dir, 'label_map.pbtxt')
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = os.path.join(prm.annotations_dir, 'eval.record')
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(output_file, 'wb') as f:
        f.write(config_text)
    shutil.copy2(output_file, prm.model_dir)
    print('The train pipeline content is:')
    print(str(config_text))

if __name__ == '__main__':
    prm = ('prm' in locals() and prm) or TrainParameters.default
    config_train_pipeline(prm)

#@markdown ---

# Commented out IPython magic to ensure Python compatibility.
# Module: train_tensorboard
#@title #Start the TensorBoard { vertical-output: true }
#@markdown The TensorBoard is run for checking the progress.
#@markdown
#@markdown Click the refresh button for start sampling.

def start_tensorboard(prm: TrainParameters):
#     %load_ext tensorboard
#     %tensorboard --logdir {os.path.join(prm.model_dir, 'train')}
    pass

if __name__ == '__main__':
    prm = ('prm' in locals() and prm) or BaseParameters.default
    start_tensorboard(prm)

#@markdown ---

# Module: train.py
#@title #Train
#@markdown The main train loop. It trains the model and put it in the output directory.
#@markdown 
#@markdown It can be stopped before the completion when
#@markdown a considerable result is reached and restart after for enhancing the tuning.


import  os
import  sys
from    absl import flags

try:
    from    utilities import execute, execute_colab, execute_non_colab
except: pass

flags.DEFINE_string('model_type', None, 'Type of the base model.')
flags.DEFINE_string('train_images_dir', None, 'Path to the directory '
                    'containing the images for train and their labeling xml.')
flags.DEFINE_string('eval_images_dir', None, 'Path to the directory '
                    'containing the images for evaluate and their labeling xml.')
FLAGS = flags.FLAGS

def main(unused_argv):
    # Part of code not executed on Colab notebook
    def run_py_mode():
        # Init the train environment
        from pretrained_model import download_pretrained_model
        from tf_record import create_tf_records
        from train_environment import init_train_environment
        from train_parameters import TrainParameters
        from train_pipeline import config_train_pipeline
        prm = TrainParameters()
        prm.update_values()
        init_train_environment(prm)
        download_pretrained_model(prm)
        create_tf_records(prm)
        config_train_pipeline(prm)
        # Import the train main function
        from object_detection import model_main_tf2
        prm.update_flags()
        # Start the tensorboard
        import subprocess
        try:
            subprocess.Popen(
                ['tensorboard', '--logdir',
                 os.path.join(prm.model_dir, 'train')],
                 stdout = subprocess.PIPE,
                 universal_newlines = True)
        except:
            try:
                tensorboard_path = os.path.join(os.path.dirname(sys.executable), 'tensorboard')
                subprocess.Popen(
                    [tensorboard_path, '--logdir',
                     os.path.join(prm.model_dir, 'train')],
                     stdout = subprocess.PIPE,
                     universal_newlines = True)
            except:
                print('Warning: cannot start tensorboard')
        # Execute the train
        model_main_tf2.main(unused_argv)
    def run_notebook_mode():
        # Import the train main function
        from object_detection import model_main_tf2
        prm.update_flags()
        # Execute the train
        model_main_tf2.main(unused_argv)
    # Execution
    execute_non_colab(run_py_mode)
    execute_colab(run_notebook_mode)

if __name__ == '__main__':
    def init():
        from od_install import install_object_detection
        install_object_detection()
    execute_non_colab(init)
    import tensorflow as tf
    tf.compat.v1.app.run()

#@markdown ---