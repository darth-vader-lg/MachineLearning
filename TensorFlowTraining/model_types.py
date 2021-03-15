# Module: model_types.py
#@title #Model types { vertical-output: true, form-width: "20%" }
#@markdown Initialize the list of the available pre-trained models and their parameters.

""" List of the available models and their definitions """
models = {
    'CenterNet Resnet101 V1 FPN 512x512': {
        'dir_name': 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 512,
        'width': 512
    },
    'SSD MobileNet v2 320x320': {
        'dir_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 300,
        'width': 300
    },
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)': {
        'dir_name': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
}

if __name__ == '__main__':
    import pprint
    pprint.PrettyPrinter(1).pprint(models)
    print('Dictionary of pre-trained models configured')

#@markdown ---
