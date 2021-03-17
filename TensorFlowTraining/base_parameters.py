# Module: base_parameters.py
#@title #Base parameters { form-width: "20%" }
#@markdown Definition of the base parameters class.

from    absl import flags
import  os
from    pathlib import Path
import  sys
import  tempfile

try:    from    default_cfg import *
except: pass
try:    from    model_types import models
except: pass

class BaseParameters:
    """ Class holding the base parameters """
    def __init__(self):
        """ Constructor """
        self._model_type = cfg_model_type or 'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)'
        self._model_dir = cfg_trained_model or 'trained-model'
        self._train_images_dir = cfg_train_images_dir or 'images/train'
        self._eval_images_dir = cfg_eval_images_dir or 'images/eval'
        self._annotations_dir = 'annotations'
        self._pre_trained_model_base_dir = os.path.join(tempfile.gettempdir(), 'tensorflow-pre-trained-models')
        self._is_path = [
            'model_dir',
            'train_images_dir',
            'eval_images_dir',
            'annotations_dir']
        self._flags_aliases = {}
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
                    if (prop in self._flags_aliases):
                        prop = self._flags_aliases[prop]
                    setattr(flags.FLAGS, prop, value)
                    print(f'Written flag {prop} with value {value}')
            except:
                pass
    def update_values(self):
        propnames = [p for p in dir(type(self)) if isinstance(getattr(type(self), p),property)]
        for prop in propnames:
            try:
                if (prop in self._flags_aliases):
                    flag = self._flags_aliases[prop]
                else:
                    flag = prop
                value = getattr(flags.FLAGS, flag)
                if (value):
                    setattr(self, prop, value)
                    print(f'Written property {prop} with value {value}')
            except:
                pass

BaseParameters.default = BaseParameters.default or BaseParameters()

if __name__ == '__main__':
    prm = ('prm' in locals() and isinstance(prm, BaseParameters) and prm) or BaseParameters.default
    print(prm)
    print('Base parameters configured')

#@markdown ---
