# Module: model_types.py
#@title #Model types { vertical-output: true, form-width: "20%" }
#@markdown Initialize the list of the available pre-trained models and their parameters.

""" List of the available models and their definitions """
models = {
    'CenterNet HourGlass104 512x512': {
        'dir_name': 'centernet_hg104_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 512,
        'width': 512
    },
    'CenterNet HourGlass104 1024x1024': {
        'dir_name': 'centernet_hg104_1024x1024_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'CenterNet Resnet50 V1 FPN 512x512': {
        'dir_name': 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 512,
        'width': 512
    },
    'CenterNet Resnet101 V1 FPN 512x512': {
        'dir_name': 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 512,
        'width': 512
    },
    'CenterNet Resnet50 V2 512x512': {
        'dir_name': 'centernet_resnet50_v2_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 512,
        'width': 512
    },
    'CenterNet MobileNetV2 FPN 512x512': {
        'dir_name': 'CenterNet MobileNetV2 FPN 512x512.tar',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz',
        'batch_size': 8,
        'height': 512,
        'width': 512
    },
    'EfficientDet D0 512x512': {
        'dir_name': 'efficientdet_d0_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 512,
        'width': 512
    },
    'EfficientDet D1 640x640': {
        'dir_name': 'efficientdet_d1_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'EfficientDet D2 768x768': {
        'dir_name': 'efficientdet_d2_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 768,
        'width': 768
    },
    'EfficientDet D3 896x896': {
        'dir_name': 'efficientdet_d3_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 896,
        'width': 896
    },
    'EfficientDet D4 1024x1024': {
        'dir_name': 'efficientdet_d4_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'EfficientDet D5 1280x1280': {
        'dir_name': 'efficientdet_d5_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 1280,
        'width': 1280
    },
    'EfficientDet D6 1280x1280': { # Really speaking it's 1408
        'dir_name': 'efficientdet_d6_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 1408,
        'width': 1408
    },
    'EfficientDet D7 1536x1536': {
        'dir_name': 'efficientdet_d7_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz',
        'batch_size': 8,
        'height': 1536,
        'width': 1536
    },
    'SSD MobileNet v2 320x320': { # Really speaking it's 300
        'dir_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 16,
        'height': 300,
        'width': 300
    },
    'SSD MobileNet V1 FPN 640x640': {
        'dir_name': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16,
        'height': 640,
        'width': 640
    },
    'SSD MobileNet V2 FPNLite 320x320': {
        'dir_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 16,
        'height': 320,
        'width': 320
    },
    'SSD MobileNet V2 FPNLite 640x640': {
        'dir_name': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16,
        'height': 640,
        'width': 640
    },
    'SSD MobileNet V2 FPNLite 640x640': {
        'dir_name': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 16,
        'height': 640,
        'width': 640
    },
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)': {
        'dir_name': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)': {
        'dir_name': 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)': {
        'dir_name': 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)': {
        'dir_name': 'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)': {
        'dir_name': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)': {
        'dir_name': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'Faster R-CNN ResNet50 V1 640x640': {
        'dir_name': 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'Faster R-CNN ResNet50 V1 1024x1024': {
        'dir_name': 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'Faster R-CNN ResNet50 V1 800x1333': {
        'dir_name': 'faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz',
        'batch_size': 8,
        'height': 800,
        'width': 1333
    },
    'Faster R-CNN ResNet101 V1 640x640': {
        'dir_name': 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'Faster R-CNN ResNet101 V1 1024x1024': {
        'dir_name': 'faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'Faster R-CNN ResNet101 V1 800x1333': {
        'dir_name': 'faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz',
        'batch_size': 8,
        'height': 800,
        'width': 1333
    },
    'Faster R-CNN ResNet152 V1 640x640': {
        'dir_name': 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'Faster R-CNN ResNet152 V1 1024x1024': {
        'dir_name': 'faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
    'Faster R-CNN ResNet152 V1 800x1333': {
        'dir_name': 'faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz',
        'batch_size': 8,
        'height': 800,
        'width': 1333
    },
    'Faster R-CNN Inception ResNet V2 640x640': {
        'dir_name': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 640,
        'width': 640
    },
    'Faster R-CNN Inception ResNet V2 1024x1024': { # Really speaking it's 800x1333
        'dir_name': 'faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz',
        'batch_size': 8,
        'height': 800,
        'width': 1333
    },
    'Mask R-CNN Inception ResNet V2 1024x1024': {
        'dir_name': 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz',
        'batch_size': 8,
        'height': 1024,
        'width': 1024
    },
}

if __name__ == '__main__':
    import pprint
    pprint.PrettyPrinter(1).pprint(models)
    print('Dictionary of pre-trained models configured')

#@markdown ---
