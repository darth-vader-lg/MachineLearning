# Module: tf_record.py

import  glob
import  io
import  os
from    pathlib import Path
import  shutil

try:
    from    base_parameters import BaseParameters
except: pass

class TFRecord:
    """Class for the TensorFlow records creation"""
    def __init__(self):
        """ Constructor """
        super().__init__()
        self._label_set = set()
        self._label_dict = dict()
    def class_text_to_int(self, row_label):
        """
        Convertion of the text of the labels to an integer index
        Keyword arguments:
        row_label   -- the label to convert to int
        """
        if (len(self._label_dict) == 0):
            count = len(self._label_set)
            labelIx = 1
            for label in self._label_set:
                self._label_dict[label] = labelIx
                labelIx += 1
        return self._label_dict[row_label]
    def create_tf_example(self, group, path):
        """
        TensorFlow example creator
        Keyword arguments:
        group   -- group's name
        path    -- path of the labeled images
        """
        from object_detection.utils import dataset_util
        from PIL import Image
        import tensorflow as tf
        with tf.compat.v1.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
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
            classes.append(self.class_text_to_int(row['class']))
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
    def create_tf_record(self, image_dir, output_file, labels_file = None, csv_file = None):
        """
        TensorFlow record creator
        Keyword arguments:
        image_dir   -- the directory containing the images
        output_file -- the output file path and name
        labels_file -- the optional output file path and name of the resulting labels file
        csv_file    -- the optional output file path and name of the csv file
        """
        import tensorflow as tf
        writer = tf.compat.v1.python_io.TFRecordWriter(output_file)
        path = os.path.join(image_dir)
        examples = self.xml_to_csv(image_dir)
        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print(f'Created the TFRecord file {str(Path(output_file).resolve())}')
        if labels_file is not None:
            from google.protobuf import text_format
            from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
            msg = StringIntLabelMap()
            for id, name in enumerate(self._label_set, start = 1):
                msg.item.append(StringIntLabelMapItem(id = id, name = name))
            text = str(text_format.MessageToBytes(msg, as_utf8 = True), 'utf-8')
            with open(labels_file, 'w') as f:
                f.write(text)
            print(f'Created the labels map file {str(Path(labels_file).resolve())}')
        if csv_file is not None:
            examples.to_csv(csv_file, index = None)
            print(f'Created the CSV file {str(Path(csv_file).resolve())}')
    def split(self, df, group):
        """
        Split the labels in an image
        Keyword arguments:
        df      -- TensorFlow example
        group   -- group's name
        """
        from collections import namedtuple
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
    def xml_to_csv(self, path):
        """
        Convert the xml files generated by labeling image softwares into the cvs panda format
        Keyword arguments:
        path    -- Path of the generated csv file
        """
        import pandas as pd
        import xml.etree.ElementTree as ET
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (
                    root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text))
                xml_list.append(value)
                self._label_set.add(member[0].text)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns = column_name)
        return xml_df

def create_tf_records(prm: BaseParameters):
    """
    TensorFlow record files creator
    Keyword arguments:
    prm     -- Parameters
    """
    print("Creating TFRecord for the train images...")
    TFRecord().create_tf_record(
        prm.train_images_dir,
        os.path.join(prm.annotations_dir, 'train.record'),
        os.path.join(prm.annotations_dir, 'label_map.pbtxt'))
    print("Creating TFRecord for the evaluation images...")
    TFRecord().create_tf_record(
        prm.eval_images_dir,
        os.path.join(prm.annotations_dir, 'eval.record'))
    shutil.copy2(os.path.join(prm.annotations_dir, 'label_map.pbtxt'), prm.model_dir)
    print(f"The labels map file was copied to {(os.path.join(str(Path(prm.model_dir).resolve()), 'label_map.pbtxt'))}")

if __name__ == '__main__':
    prm = ('prm' in locals() and prm) or BaseParameters.default
    create_tf_records(prm)
