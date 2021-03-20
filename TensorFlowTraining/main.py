# Module: main.py
# Main script for training and exporting a model

try:    from    utilities import *
except: pass

def main(unused_argv):
    try:
        print('=' * 80)
        print('Train')
        from    train_main import train_main
        train_main(unused_argv)
    except (KeyboardInterrupt, SystemExit):
        pass
    try:
        print('=' * 80)
        print('=' * 80)
        print('=' * 80)
        print('=' * 80)
        print('Export to pb SavedModel')
        from    export_main import export_main
        export_main(unused_argv)
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == '__main__':
    if (not is_jupyter()):
        from od_install import install_object_detection
        install_object_detection()
    try:
        import tensorflow as tf
        tf.compat.v1.app.run(main)
    except KeyboardInterrupt:
        print('Interrupted by user')
    except SystemExit:
        print('End')
