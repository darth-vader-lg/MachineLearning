# Module base_parameters.py

from model_types import models

# Begin notebook cell
import  os
from    pathlib import Path
import  tempfile
import  sys

class BaseParameters:
    """ Class holding the base parameters """
    def __init__(self):
        """ Constructor """
        self._model_type = 'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)'
        self._model_dir = 'trained-model'
        self._train_images_dir = 'images/train'
        self._eval_images_dir = 'images/eval'
    @property
    def model_type(self): return flags.FLAGS.model_type or self._model_type
    @property
    def model_dir(self): return flags.FLAGS.model_dir or self._model_dir
    @property
    def train_images_dir(self): return flags.FLAGS.train_images_dir or self._train_images_dir
    @property
    def eval_images_dir(self): return flags.FLAGS.eval_images_dir or self._eval_images_dir
    @property
    def annotations_dir(self): return 'annotations'
    @property
    def model(self):
        global models
        return models[self.model_type]
    @property
    def pre_trained_model_base_dir(self): return os.path.join(tempfile.gettempdir(), "tensorflow-pre-trained-models")
    @property
    def pre_trained_model_dir(self):
        return str(Path(os.path.join(self.pre_trained_model_base_dir, Path(self.model["DownloadPath"]).name)).with_suffix("").with_suffix(""))

""" Arguments definition """
from absl import flags
flags.DEFINE_string(
    'model_type',
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)',
    'Type of the base model.')
flags.DEFINE_string(
    'model_dir',
    'trained-model',
    'Path to output model directory where event and checkpoint files will be written.')
flags.DEFINE_string(
    'train_images_dir',
    'images/train',
    'Path to the directory containing the images for train and their labeling xml.')
flags.DEFINE_string(
    'eval_images_dir',
    'images/eval',
    'Path to the directory containing the images for evaluate and their labeling xml.')
# End notebook cell
