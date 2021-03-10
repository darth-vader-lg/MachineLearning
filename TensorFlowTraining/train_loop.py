# Module train_loop.py

from    od_install import install_object_detection
from    pretrained_model import download_pretrained_model
from    process_execution import execute_non_colab
from    tf_record import create_tf_records
from    train_parameters import TrainParameters
from    train_pipeline import config_train_pipeline
from    train_environment import init_train_environment

from    train_parameters import TrainParameters
from    train_parameters import prm

# Begin notebook cell
import  os

def train_loop(_):
    """
    Train loop function
    """
    import tensorflow as tf
    from object_detection import model_lib_v2
    if prm.checkpoint_dir:
        model_lib_v2.eval_continuously(
            pipeline_config_path = os.path.join(prm.annotations_dir, "pipeline.config"),
            model_dir = prm.num_train_steps,
            sample_1_of_n_eval_examples = prm.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples = prm.sample_1_of_n_eval_on_train_examples,
            checkpoint_dir = prm.checkpoint_dir,
            wait_interval = 300,
            timeout = prm.eval_timeout)
    else:
        if prm.use_tpu:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platform.
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(prm.tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        elif prm.num_workers > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path = os.path.join(prm.annotations_dir, "pipeline.config"),
            model_dir = prm.model_dir,
            train_steps = prm.num_train_steps,
            use_tpu = prm.use_tpu,
            checkpoint_every_n = prm.checkpoint_every_n,
            record_summaries = prm.record_summaries)

if __name__ == '__main__':
    import tensorflow as tf
    tf.compat.v1.app.run(train_loop)
# End notebook cell
