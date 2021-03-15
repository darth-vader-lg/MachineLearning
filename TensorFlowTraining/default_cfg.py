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
#@markdown ## Target directory:
#@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where the checkpoints will be saved.
cfg_trained_model = 'trained-model' #@param {type:"string"}
#@markdown ---
#@markdown ## Images directories:
#@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where is located the images set for the train and for the evaluation.
cfg_train_images_dir = 'images/train' #@param {type:"string"}
cfg_eval_images_dir = 'images/eval' #@param {type:"string"}
#@markdown ---
