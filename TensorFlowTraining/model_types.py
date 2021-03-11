# Module: model_types.py

""" List of the available models and their definitions """
models = {
    'SSD MobileNet v2 320x320': {
        'DownloadPath': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 12,
        'height': 300,
        'width': 300
    },
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)': {
        'DownloadPath': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
}

if __name__ == '__main__':
    import pprint
    pprint.PrettyPrinter(1).pprint(models)
    print('Dictionary of pre-trained models configured')
