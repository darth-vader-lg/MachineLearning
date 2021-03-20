# Module: main.py
# Main script for training and exporting a model

try:    from    utilities import *
except: pass

def main(unused_argv):
    from    train_main import train_main
    train_main(unused_argv)
    from    export_main import export_main
    export_main(unused_argv)

if __name__ == '__main__':
    if (not is_jupyter()):
        from od_install import install_object_detection
        install_object_detection()
    import tensorflow as tf
    tf.compat.v1.app.run(main)
