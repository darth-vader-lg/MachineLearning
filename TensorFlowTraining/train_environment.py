# Module train_environment.py

from    train_parameters import TrainParameters

# Begin notebook cell
import  os
import  sys
from    absl import app

def init_train_environment(prm = TrainParameters()):
    """
    Initialize the training environment with the right directories structure.
    Keyword arguments:
    prm     -- the train parameters
    """
    # Set the configuration for Google Colab
    if ('google.colab' in sys.modules):
        if (not os.path.exists("/mnt/MyDrive")):
            print("Mounting the GDrive")
            from google.colab import drive
            drive.mount("/mnt")
        # Check the existence of the train images dir
        gdriveOutputDir = os.path.join("/mnt", "MyDrive", prm.train_images_dir)
        if (not os.path.isdir(gdriveOutputDir)):
            raise Exception("Error!!! The train images dir doesn't exist")
        if (os.path.exists("/content/train-images")):
            os.unlink("/content/train-images")
        os.symlink(gdriveOutputDir, "/content/train-images", True)
        # Check the existence of the evaluation images dir
        gdriveOutputDir = os.path.join("/mnt", "MyDrive", prm.eval_images_dir)
        if (not os.path.isdir(gdriveOutputDir)):
            raise Exception("Error!!! The evaluation images dir doesn't exist")
        if (os.path.exists("/content/eval-images")):
            os.unlink("/content/eval-images")
        os.symlink(gdriveOutputDir, "/content/eval-images", True)
        # Check the existence of the output directory
        gdriveOutputDir = os.path.join("/mnt", "MyDrive", prm.model_dir)
        if (not os.path.isdir(gdriveOutputDir)):
            print("Creating the output directory")
            os.mkdir(gdriveOutputDir)
        if (os.path.exists("/content/trained-model")):
            os.unlink("/content/trained-model")
        os.symlink(gdriveOutputDir, "/content/trained-model", True)
    else:
        if (not os.path.isdir(prm.train_images_dir)):
            raise Exception("Error!!! The train images dir doesn't exist")
        if (not os.path.isdir(prm.eval_images_dir)):
            raise Exception("Error!!! The evaluation images dir doesn't exist")
        if (not os.path.exists(prm.model_dir)):
            print("Creating the output dir")
            os.mkdir(prm.model_dir)
    if (not os.path.exists(prm.annotations_dir)):
        os.mkdir(prm.annotations_dir)

if __name__ == '__main__':
    sys.argv = sys.argv[:1]
    try:
        app.run(lambda argv: None)
    except:
        pass
    init_train_environment()
# End notebook cell
