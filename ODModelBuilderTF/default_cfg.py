class Cfg(object):
    #@markdown ## Data on Google Drive:
    #@markdown (The data will be treated in a Google Drive space if enabled)
    data_on_drive = True #@param {type:"boolean"}
    #@markdown ---
    #@markdown ## Base model:
    #@markdown (The base model from which the train will start)
    model_type = 'SSD MobileNet v2 320x320' #@param ['CenterNet HourGlass104 512x512', 'CenterNet HourGlass104 1024x1024', 'CenterNet Resnet50 V1 FPN 512x512', 'CenterNet Resnet101 V1 FPN 512x512', 'CenterNet Resnet50 V2 512x512', 'CenterNet MobileNetV2 FPN 512x512', 'EfficientDet D0 512x512', 'EfficientDet D1 640x640', 'EfficientDet D2 768x768', 'EfficientDet D3 896x896', 'EfficientDet D4 1024x1024', 'EfficientDet D5 1280x1280', 'EfficientDet D6 1280x1280', 'EfficientDet D7 1536x1536', 'SSD MobileNet v2 320x320', 'SSD MobileNet V1 FPN 640x640', 'SSD MobileNet V2 FPNLite 320x320', 'SSD MobileNet V2 FPNLite 640x640', 'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)', 'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)', 'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)', 'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)', 'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)', 'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)', 'Faster R-CNN ResNet50 V1 640x640', 'Faster R-CNN ResNet50 V1 1024x1024', 'Faster R-CNN ResNet50 V1 800x1333', 'Faster R-CNN ResNet101 V1 640x640', 'Faster R-CNN ResNet101 V1 1024x1024', 'Faster R-CNN ResNet101 V1 800x1333', 'Faster R-CNN ResNet152 V1 640x640', 'Faster R-CNN ResNet152 V1 1024x1024', 'Faster R-CNN ResNet152 V1 800x1333', 'Faster R-CNN Inception ResNet V2 640x640', 'Faster R-CNN Inception ResNet V2 1024x1024', 'Mask R-CNN Inception ResNet V2 1024x1024']
    #@markdown ---
    #@markdown ## Images directories:
    #@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where is located the images set for the train and for the evaluation.
    train_images_dir = 'images/train' #@param {type:"string"}
    eval_images_dir = 'images/eval' #@param {type:"string"}
    #@markdown ---
    #@markdown ## Train directory:
    #@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where the checkpoints will be saved.
    trained_model = 'trained-model' #@param {type:"string"}
    #@markdown ---
    #@markdown ## Export directory:
    #@markdown The GDrive directory (Colab execution) or the local directory (machine execution) where the exported model will be saved.
    exported_model = 'exported-model' #@param {type:"string"}
    #@markdown ---
    #@markdown ## Maximum training steps:
    #@markdown The maximun number of train steps. If < 0 it will be limited by the base model configuration.
    max_train_steps = -1 #@param {type:"integer"}
    #@markdown ---
    #@markdown ## Batch size:
    #@markdown The size of the batch. If < 1 the value contained in the model pipeline configuration will be used
    batch_size = 16 #@param {type:"integer"}
    #@markdown ---
    # TensorFlow version
    tensorflow_version = 'tensorflow==2.4.1' # or for example tf-nightly==2.5.0.dev20210315
    # SHA1 for the checkout of the TensorFlow object detection api
    od_api_git_sha1 = 'e356598a5b79a768942168b10d9c1acaa923bdb4'
