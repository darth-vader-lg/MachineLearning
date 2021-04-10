import  os
import  shutil

try:    from    default_cfg import Cfg
except: pass
try:    from    train_parameters import TrainParameters
except: pass

def config_train_pipeline(prm: TrainParameters):
    """
    Configure the training pipeline
    Keyword arguments:
    prm     -- Parameters
    """
    import  tensorflow as tf
    from    object_detection.protos import pipeline_pb2
    from    object_detection.utils import label_map_util
    from    google.protobuf import text_format
    import  tempfile
    # Copy the pipeline configuration file if it's not already present in the output dir
    print('Configuring the pipeline')
    output_file = prm.pipeline_config_path
    pre_trained_model_dir = os.path.join(prm.pre_trained_model_base_dir, prm.model['dir_name'])
    pre_trained_cfg_file = os.path.join(
        tempfile.gettempdir(),
        'tf-od-api-' + Cfg.od_api_git_sha1,
        'research', 'object_detection', 'configs', 'tf2',
        prm.model['dir_name'] + '.config')
    shutil.copy2(pre_trained_cfg_file, output_file)
    # Read the number of labels
    label_dict = label_map_util.get_label_map_dict(os.path.join(prm.annotations_dir, 'label_map.pbtxt'))
    labels_count = len(label_dict)
    # Configuring the pipeline
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(output_file, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    pipeline_config.model.ssd.num_classes = labels_count
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = prm.model['height']
    pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = prm.model['width']
    pipeline_config.train_config.batch_size = prm.batch_size if prm.batch_size > 0 else pipeline_config.train_config.batch_size
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(pre_trained_model_dir, 'checkpoint', 'ckpt-0')
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
    prm = ('prm' in locals() and isinstance(prm, TrainParameters) and prm) or TrainParameters.default
    config_train_pipeline(prm)
