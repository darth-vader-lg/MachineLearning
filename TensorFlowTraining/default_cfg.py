# Module: default_cfg.py
#@title #Notebook configuration
#@markdown ## Data on Google Drive:
#@markdown (The data will be treated in a Google Drive space if enabled)
cfg_data_on_drive = True #@param {type:"boolean"}
#@markdown ---
#@markdown ## Base model:
#@markdown (The base model from which the train will start)
cfg_model_type = 'SSD MobileNet v2 320x320' #@param ['SSD MobileNet v2 320x320', 'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)']
#@markdown ---
#@markdown ## Images directories:
#@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where is located the images set for the train and for the evaluation.
cfg_train_images_dir = 'images/train' #@param {type:"string"}
cfg_eval_images_dir = 'images/eval' #@param {type:"string"}
#@markdown ---
#@markdown ## Train directory:
#@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where the checkpoints will be saved.
cfg_trained_model = 'trained-model' #@param {type:"string"}
#@markdown ---
#@markdown ## Export directory:
#@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where the exported model will be saved.
cfg_exported_model = 'exported-model' #@param {type:"string"}
#@markdown ---
#@markdown ---
#@markdown ## Maximum training steps:
#@markdown The maximun number of train steps. If < 0 it will be limited by the base model configuration.
cfg_max_train_steps = -1 #@param {type:"integer"}
#@markdown ---
# TensorFlow version
cfg_tensorflow_version = 'tensorflow==2.4.1' # or for example tf-nightly==2.5.0.dev20210315
# SHA1 for the checkout of the TensorFlow object detection api
cfg_od_api_git_sha1 = 'e356598a5b79a768942168b10d9c1acaa923bdb4'
