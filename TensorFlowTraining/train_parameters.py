#module train_parameters.py
#@title #Train parameters { form-width: "20%" }
#@markdown Definition of the train parameters. Read the comments in the flags
#@markdown section of the train main module
#@markdown https://raw.githubusercontent.com/tensorflow/models/e356598a5b79a768942168b10d9c1acaa923bdb4/research/object_detection/model_main_tf2.py

import  os

try:    from    base_parameters import BaseParameters
except: pass
try:    from    default_cfg import *
except: pass

class TrainParameters(BaseParameters):
    """ Class holding the train execution parameters """
    def __init__(self):
        """ Constructor """
        super().__init__()
        self._pipeline_config_path = os.path.join(self.annotations_dir, 'pipeline.config')
        self._num_train_steps = cfg_max_train_steps if cfg_max_train_steps > -1 else None
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
    prm = ('prm' in locals() and isinstance(prm, TrainParameters) and prm) or TrainParameters.default
    print(prm)
    print('Train parameters configured')

#@markdown ---
