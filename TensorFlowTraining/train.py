# Module: train.py

import  os
import  sys
from    absl import flags

try:
    from    utilities import execute_colab, execute_non_colab
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
        from train_parameters import TrainParameters
        from train_pipeline import config_train_pipeline
        from train_environment import init_train_environment
        prm = TrainParameters()
        prm.update_values()
        init_train_environment(prm)
        download_pretrained_model(prm)
        create_tf_records(prm)
        config_train_pipeline(prm)
        # Import the train main function
        from object_detection import model_main_tf2
        prm.update_flags()
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
