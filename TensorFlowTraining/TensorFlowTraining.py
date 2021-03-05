# =============================================================================
# Parameters
# =============================================================================
# Training model output directory
outputDir = "TrainedModel"
# Directory containing the images for the training
trainImagesDir = "Images/Train"
# Directory containing the images for the evaluation
testImagesDir = "Images/Test"
# The type of the base model
modelType = "SSD MobileNet v2 320x320"
# =============================================================================
# List of available models and theirs configurations
# =============================================================================
models = {
    "SSD MobileNet v2 320x320": {
        "DownloadPath": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
        "batch_size": 12,
        "height": 300,
        "width": 300
    },
}
model = models[modelType]
# =============================================================================
# Standard imports
# =============================================================================
import os
import glob
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
# =============================================================================
# Directories mounting, checking and creating
# =============================================================================
# Set the directory where to download the pre-trained models
preTrainedModelBaseDir = os.path.join(tempfile.gettempdir(), "tensorflow-pre-trained-models")
# Set the configuration for Google Colab
if ('google.colab' in sys.modules):
    if (not os.path.exists("/mnt/MyDrive")):
        print("Mounting the GDrive")
        from google.colab import drive
        drive.mount("/mnt")
    # Check the existence of the train images dir
    gdriveOutputDir = os.path.join("/mnt", "MyDrive", trainImagesDir)
    if (not os.path.isdir(gdriveOutputDir)):
        print("Error!!! The train images dir doesn't exist")
        exit(-1)
    if (not os.path.exists("/content/train-images")):
        os.mkdir("/content/train-images")
    if (os.path.exists("/content/train-images/train")):
        os.unlink("/content/train-images/train")
    os.symlink(gdriveOutputDir, "/content/train-images/train", True)
    trainImagesDir = "/content/train-images/train"
    # Check the existence of the test images dir
    gdriveOutputDir = os.path.join("/mnt", "MyDrive", testImagesDir)
    if (not os.path.isdir(gdriveOutputDir)):
        print("Error!!! The test images dir doesn't exist")
        exit(-1)
    if (os.path.exists("/content/train-images/test")):
        os.unlink("/content/train-images/test")
    os.symlink(gdriveOutputDir, "/content/train-images/test", True)
    testImagesDir = "/content/train-images/test"
    # Check the existence of the output directory
    gdriveOutputDir = os.path.join("/mnt", "MyDrive", outputDir)
    if (not os.path.isdir(gdriveOutputDir)):
        print("Creating the output directory")
        os.mkdir(gdriveOutputDir)
    if (os.path.exists("/content/trained-model")):
        os.unlink("/content/trained-model")
    os.symlink(gdriveOutputDir, "/content/trained-model", True)
    outputDir = "/content/trained-model"
else:
    if (not os.path.isdir(trainImagesDir)):
        print("Error!!! The train images dir doesn't exist")
        exit(-1)
    if (not os.path.isdir(testImagesDir)):
        print("Error!!! The test images dir doesn't exist")
        exit(-1)
    if (not os.path.exists(outputDir)):
        print("Creating the output dir")
        os.mkdir(outputDir)
if (not os.path.exists(os.path.join(outputDir, "annotations"))):
    os.mkdir(os.path.join(outputDir, "annotations"))
# =============================================================================
# Processes execution with printing
# =============================================================================
# Execute a subprocess
def executeSubprocess(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
# Execute a subprocess with output
def execute(cmd):
    for output in executeSubprocess(cmd):
        print(output, end="")
# =============================================================================
# Well known environment installation
# =============================================================================
# Path of the python interpreter executable
pythonPath = os.path.join(os.path.dirname(sys.executable), "python3")
if (not os.path.exists(pythonPath)):
    pythonPath = os.path.join(os.path.dirname(sys.executable), "python")
# Upgrade pip and setuptools
execute([pythonPath, "-m", "pip", "install", "--upgrade", "pip==21.0.1"])
execute([pythonPath, "-m", "pip", "install", "--upgrade", "setuptools==54.0.0"])
# Install TensorFlow
execute([pythonPath, "-m", "pip", "install", "tensorflow==2.4.1"])
# Install pygit2
execute([pythonPath, "-m", "pip", "install", "pygit2==1.5.0"])
# Progress class for the git output
import pygit2
import datetime
class GitCallbacks(pygit2.RemoteCallbacks):
    def __init__(self, credentials=None, certificate=None):
        self.dateTime = datetime.datetime.now()
        return super().__init__(credentials=credentials, certificate=certificate)
    def transfer_progress(self, stats):
        now = datetime.datetime.now()
        if ((now - self.dateTime).total_seconds() > 1):
            print("\rReceiving... Deltas [%d / %d], Objects [%d / %d]"%(stats.indexed_deltas, stats.total_deltas, stats.indexed_objects, stats.total_objects), end="", flush=True)
            self.dateTime = now
        if (stats.received_objects >= stats.total_objects and stats.indexed_objects >= stats.total_objects and stats.indexed_deltas >= stats.total_deltas):
            print("\r\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\rDone Deltas %d, Objects %d."%(stats.total_objects, stats.total_objects))
        return super().transfer_progress(stats)
# Directory of the TensorFlow object detection api
odApiDir = os.path.join(tempfile.gettempdir(), "tensorflow-object-detection-api")
# Install the TensorFlow models
if (not os.path.isdir(odApiDir)):
    # Create the callback for the progress
    callbacks = GitCallbacks();
    # Clone the TensorFlow models repository
    print("Cloning the TensorFlow models repository")
    pygit2.clone_repository("https://github.com/tensorflow/models.git", odApiDir, callbacks = callbacks)
    print("TensorFlow models repository cloned")
    # Checkout a well known commit
    repo = pygit2.Repository(odApiDir)
    ish = "e356598a5b79a768942168b10d9c1acaa923bdb4"
    (commit, reference) = repo.resolve_refish(ish)
    repo.checkout_tree(commit)
    repo.reset(pygit2.Oid(hex=ish), pygit2.GIT_RESET_HARD)
    # Move to the research dir
    currentDir = os.getcwd()
    os.chdir(os.path.join(odApiDir, "research"))
    # Install the protobuf tools
    execute([pythonPath, "-m", "pip", "install", "grpcio-tools==1.32.0"])
    # Compile the protobufs
    import grpc_tools.protoc as protoc
    protoFiles = Path("object_detection/protos").rglob("*.proto")
    for protoFile in protoFiles:
        protoFilePath = str(protoFile)
        print("Compiling", protoFilePath)
        protoc.main(["grpc_tools.protoc", "--python_out=.", protoFilePath])
    # Install the object detection packages
    shutil.copy2("object_detection/packages/tf2/setup.py", ".")
    execute([pythonPath, "-m", "pip", "install", "."])
    os.chdir(currentDir)
print("Installation completed.")
# =============================================================================
# Add the object detection api to the paths
# =============================================================================
# Append the object detection api to the path
sys.path.append(os.path.join(odApiDir, "research"))
sys.path.append(os.path.join(odApiDir, "research/slim"))
# =============================================================================
# Download the base model from the TensorFlow models zoo if it's needed
# =============================================================================
# Pre-trained model download
import urllib.request
preTrainedModelDir = str(Path(os.path.join(preTrainedModelBaseDir, Path(model["DownloadPath"]).name)).with_suffix("").with_suffix(""))
if not (os.path.exists(preTrainedModelDir)):
    if (not os.path.exists(preTrainedModelBaseDir)):
        os.mkdir(preTrainedModelBaseDir)
    preTrainedModelFile = preTrainedModelDir + ".tar.gz"
    print(f"Downloading the pre-trained model {str(Path(preTrainedModelFile).name)}...")
    import tarfile
    urllib.request.urlretrieve(model["DownloadPath"], preTrainedModelFile) # TODO: show progress
    print("Done.")
    print(f"Extracting the pre-trained model {str(Path(preTrainedModelFile).name)}...")
    tar = tarfile.open(preTrainedModelFile)
    tar.extractall(preTrainedModelBaseDir)
    tar.close()
    os.remove(preTrainedModelFile)
    print("Done.")
# =============================================================================
# Prepare the TFRecords
# =============================================================================
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import io
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import tensorflow.compat.v1 as tf1
# Convert the xml files generated by labeling image softwares into the cvs panda format
labelSet = set()
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
            labelSet.add(member[0].text)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
# Dictionary name -> index
labelDict = dict()
def class_text_to_int(row_label):
    if (len(labelDict) == 0):
        count = len(labelSet)
        labelIx = 1
        for label in labelSet:
            labelDict[label] = labelIx
            labelIx += 1
        msg = StringIntLabelMap()
        for id, name in enumerate(labelSet, start=1):
            msg.item.append(StringIntLabelMapItem(id=id, name=name))
        text = str(text_format.MessageToBytes(msg, as_utf8=True), "utf-8")
        with open(os.path.join(outputDir, "annotations", "label_map.pbtxt"), "w") as f:
            f.write(text)
    return labelDict[row_label]
# Splitting
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
# TensorFlow example creator
def create_tf_example(group, path):
    with tf1.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
# TensorFlow record file creator
def create_tf_record(imageDir, outputFile, csvFile = None):
    writer = tf1.python_io.TFRecordWriter(outputFile)
    path = os.path.join(imageDir)
    examples = xml_to_csv(imageDir)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(outputFile))
    if csvFile is not None:
        examples.to_csv(csvFile, index=None)
        print('Successfully created the CSV file: {}'.format(csvFile))
# Create record files
create_tf_record(trainImagesDir, os.path.join(outputDir, "annotations", "train.record"))
create_tf_record(testImagesDir, os.path.join(outputDir, "annotations", "test.record"))
# =============================================================================
# Configuration of the training pipeline in the model output directory
# =============================================================================
# Copy the pipeline configuration file if it's not already present in the output dir
print("Configuring the pipeline")
outPipelineFile = os.path.join(outputDir, "pipeline.config")
if (not os.path.exists(outPipelineFile)):
    shutil.copy2(os.path.join(preTrainedModelDir, "pipeline.config"), outputDir)
# Configuring the pipeline
from object_detection.protos import pipeline_pb2
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(outPipelineFile, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
pipeline_config.model.ssd.num_classes = 1 # TODO define
pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = model["height"]
pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = model["width"]
pipeline_config.train_config.batch_size = model["batch_size"]
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(preTrainedModelDir, "checkpoint", "ckpt-0")
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = os.path.join(outputDir, "annotations", "label_map.pbtxt")
pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = os.path.join(outputDir, "annotations", "train.record")
pipeline_config.eval_input_reader[0].label_map_path = os.path.join(outputDir, "annotations", "label_map.pbtxt")
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = os.path.join(outputDir, "annotations", "test.record")
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(outPipelineFile, "wb") as f:
    f.write(config_text)
print(str(config_text))
print("Done.")
