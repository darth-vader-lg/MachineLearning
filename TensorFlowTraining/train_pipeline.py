# Module: train_pipeline

import  os
import  shutil

try:
    from    train_parameters import TrainParameters
except: pass

def config_train_pipeline(prm: TrainParameters):
    """
    Configure the training pipeline
    Keyword arguments:
    prm     -- Parameters
    """
    import tensorflow as tf
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    # Copy the pipeline configuration file if it's not already present in the output dir
    print('Configuring the pipeline')
    output_file = prm.pipeline_config_path
    shutil.copy2(os.path.join(prm.pre_trained_model_dir, 'pipeline.config'), output_file)
    # Configuring the pipeline
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(output_file, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    pipeline_config.model.ssd.num_classes = 1 # TODO define
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = prm.model['height']
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = prm.model['width']
    pipeline_config.train_config.batch_size = prm.model['batch_size']
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(prm.pre_trained_model_dir, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path = os.path.join(prm.annotations_dir, 'label_map.pbtxt')
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = os.path.join(prm.annotations_dir, 'train.record')
    pipeline_config.eval_input_reader[0].label_map_path = os.path.join(prm.annotations_dir, 'label_map.pbtxt')
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = os.path.join(prm.annotations_dir, 'eval.record')
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(output_file, 'wb') as f:
        f.write(config_text)
    shutil.copy2(output_file, prm.model_dir)
    print('The train pipeline content is:')
    print(str(config_text))

if __name__ == '__main__':
    prm = prm or TrainParameters.default
    config_train_pipeline(TrainParameters.default)
