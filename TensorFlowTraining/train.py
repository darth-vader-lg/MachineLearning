# Module train.py

from    od_install import install_object_detection
import  os
from    pretrained_model import download_pretrained_model
from    process_execution import execute_non_colab
from    tf_record import create_tf_records
from    train_parameters import TrainParameters, prm
from    train_pipeline import config_train_pipeline
from    train_environment import init_train_environment
from    train_loop import train_loop

def main(unused_argv):
    """
    Main train function
    Keyword arguments:
    prm     -- Parameters
    """
    execute_non_colab(init_train_environment)
    execute_non_colab(download_pretrained_model)
    execute_non_colab(create_tf_records)
    execute_non_colab(config_train_pipeline)
    execute_non_colab(train_loop)

if __name__ == '__main__':
    execute_non_colab(install_object_detection)
    import tensorflow as tf
    if (__debug__):
        tf.config.run_functions_eagerly(True)
    tf.compat.v1.app.run()
# End notebook cell
