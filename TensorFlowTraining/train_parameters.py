# Module train_parameters.py

from    base_parameters import BaseParameters
import  model_types

# Begin notebook cell
import  sys

class TrainParameters(BaseParameters):
    """ Class holding the train execution parameters """
    def __init__(self):
        """ Constructor """
        super().__init__()
        self._num_train_steps = 10000
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
    @property
    def num_train_steps(self):
        return flags.FLAGS.num_train_steps or self._num_train_steps
    @property
    def eval_on_train_data(self):
        return flags.FLAGS.eval_on_train_data or self._eval_on_train_data
    @property
    def sample_1_of_n_eval_examples(self):
        return flags.FLAGS.sample_1_of_n_eval_examples or self._sample_1_of_n_eval_examples
    @property
    def sample_1_of_n_eval_on_train_examples(self):
        return flags.FLAGS.sample_1_of_n_eval_on_train_examples or self._sample_1_of_n_eval_on_train_examples
    @property
    def checkpoint_dir(self):
        return flags.FLAGS.checkpoint_dir or self._checkpoint_dir
    @property
    def eval_timeout(self):
        return flags.FLAGS.eval_timeout or self._eval_timeout
    @property
    def use_tpu(self):
        return flags.FLAGS.use_tpu or self._use_tpu
    @property
    def tpu_name(self):
        return flags.FLAGS.tpu_name or self._tpu_name
    @property
    def num_workers(self):
        return flags.FLAGS.num_workers or self._num_workers
    @property
    def checkpoint_every_n(self):
        return flags.FLAGS.checkpoint_every_n or self._checkpoint_every_n
    @property
    def record_summaries(self):
        return flags.FLAGS.record_summaries or self._record_summaries

""" Arguments definition """
from absl import flags
flags.DEFINE_integer(
    'num_train_steps',
    10000,
    'Number of train steps.')
flags.DEFINE_bool(
    'eval_on_train_data',
    False,
    'Enable evaluating on train data (only supported in distributed training).')
flags.DEFINE_integer(
    'sample_1_of_n_eval_examples',
    None,
    'Will sample one of every n eval input examples, where n is provided.')
flags.DEFINE_integer(
    'sample_1_of_n_eval_on_train_examples',
    5,
    'Will sample one of every n train input examples for evaluation, '
    'where n is provided. This is only used if `eval_training_data` is True.')
flags.DEFINE_string(
    'checkpoint_dir',
    None,
    'Path to directory holding a checkpoint. If `checkpoint_dir` is provided, '
    'this binary operates in eval-only mode, writing resulting metrics to `model_dir`.')
flags.DEFINE_integer(
    'eval_timeout',
    3600,
    'Number of seconds to wait for an evaluation checkpoint before exiting.')
flags.DEFINE_bool(
    'use_tpu',
    False,
    'Whether the job is executing on a TPU.')
flags.DEFINE_string(
    'tpu_name',
    None,
    'Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer(
    'num_workers',
    1,
    'When num_workers > 1, training uses MultiWorkerMirroredStrategy. '
    'When num_workers = 1 it uses MirroredStrategy.')
flags.DEFINE_integer(
    'checkpoint_every_n',
    1000,
    'Integer defining how often we checkpoint.')
flags.DEFINE_boolean(
    'record_summaries',
    True,
    'Whether or not to record summaries during training.')

prm = TrainParameters()
# End notebook cell

