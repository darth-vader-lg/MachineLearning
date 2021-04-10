""" List of the available models and their definitions """
models = {
    'CenterNet HourGlass104 512x512': {
        'dir_name': 'centernet_hg104_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz',
        'height': 512,
        'width': 512,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:3',
            'classes': 'StatefulPartitionedCall:1',
            'scores': 'StatefulPartitionedCall:2',
            'boxes': 'StatefulPartitionedCall:0',
            'start_id': 1,
        }
    },
    'CenterNet HourGlass104 1024x1024': {
        'dir_name': 'centernet_hg104_1024x1024_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:3',
            'classes': 'StatefulPartitionedCall:1',
            'scores': 'StatefulPartitionedCall:2',
            'boxes': 'StatefulPartitionedCall:0',
            'start_id': 1,
         }
    },
    'CenterNet Resnet50 V1 FPN 512x512': {
        'dir_name': 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz',
        'height': 512,
        'width': 512,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:3',
            'classes': 'StatefulPartitionedCall:1',
            'scores': 'StatefulPartitionedCall:2',
            'boxes': 'StatefulPartitionedCall:0',
            'start_id': 1,
        }
    },
    'CenterNet Resnet101 V1 FPN 512x512': {
        'dir_name': 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz',
        'height': 512,
        'width': 512,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:3',
            'classes': 'StatefulPartitionedCall:1',
            'scores': 'StatefulPartitionedCall:2',
            'boxes': 'StatefulPartitionedCall:0',
            'start_id': 1,
        }
    },
    'CenterNet Resnet50 V2 512x512': {
        'dir_name': 'centernet_resnet50_v2_512x512_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz',
        'height': 512,
        'width': 512,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:3',
            'classes': 'StatefulPartitionedCall:1',
            'scores': 'StatefulPartitionedCall:2',
            'boxes': 'StatefulPartitionedCall:0',
            'start_id': 1,
        }
    },
    'CenterNet MobileNetV2 FPN 512x512': {
        'dir_name': 'CenterNet MobileNetV2 FPN 512x512.tar',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz',
        'height': 512,
        'width': 512,
        'net_config': {
            'fixed_width': 320,
            'fixed_height': 320,
            'input_image': 'serving_default_input:0',
            'detections': 'StatefulPartitionedCall:0',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:1',
            'boxes': 'StatefulPartitionedCall:3',
            'start_id': 0,
        }
    },
    'EfficientDet D0 512x512': {
        'dir_name': 'efficientdet_d0_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz',
        'height': 512,
        'width': 512,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'EfficientDet D1 640x640': {
        'dir_name': 'efficientdet_d1_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'EfficientDet D2 768x768': {
        'dir_name': 'efficientdet_d2_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz',
        'height': 768,
        'width': 768,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'EfficientDet D3 896x896': {
        'dir_name': 'efficientdet_d3_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz',
        'height': 896,
        'width': 896,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'EfficientDet D4 1024x1024': {
        'dir_name': 'efficientdet_d4_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'EfficientDet D5 1280x1280': {
        'dir_name': 'efficientdet_d5_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz',
        'height': 1280,
        'width': 1280,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'EfficientDet D6 1280x1280': { # Really speaking it's 1408
        'dir_name': 'efficientdet_d6_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz',
        'height': 1408,
        'width': 1408,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'EfficientDet D7 1536x1536': {
        'dir_name': 'efficientdet_d7_coco17_tpu-32',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz',
        'height': 1536,
        'width': 1536,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD MobileNet v2 320x320': { # Really speaking it's 300
        'dir_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'height': 300,
        'width': 300,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD MobileNet V1 FPN 640x640': {
        'dir_name': 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD MobileNet V2 FPNLite 320x320': {
        'dir_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
        'height': 320,
        'width': 320,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD MobileNet V2 FPNLite 640x640': {
        'dir_name': 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)': {
        'dir_name': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)': {
        'dir_name': 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)': {
        'dir_name': 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)': {
        'dir_name': 'ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)': {
        'dir_name': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)': {
        'dir_name': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet50 V1 640x640': {
        'dir_name': 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet50 V1 1024x1024': {
        'dir_name': 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet50 V1 800x1333': {
        'dir_name': 'faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz',
        'height': 800,
        'width': 1333,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet101 V1 640x640': {
        'dir_name': 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet101 V1 1024x1024': {
        'dir_name': 'faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet101 V1 800x1333': {
        'dir_name': 'faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz',
        'height': 800,
        'width': 1333,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet152 V1 640x640': {
        'dir_name': 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet152 V1 1024x1024': {
        'dir_name': 'faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN ResNet152 V1 800x1333': {
        'dir_name': 'faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz',
        'height': 800,
        'width': 1333,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN Inception ResNet V2 640x640': {
        'dir_name': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz',
        'height': 640,
        'width': 640,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Faster R-CNN Inception ResNet V2 1024x1024': { # Really speaking it's 800x1333
        'dir_name': 'faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz',
        'height': 800,
        'width': 1333,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:5',
            'classes': 'StatefulPartitionedCall:2',
            'scores': 'StatefulPartitionedCall:4',
            'boxes': 'StatefulPartitionedCall:1',
            'start_id': 1,
        }
    },
    'Mask R-CNN Inception ResNet V2 1024x1024': {
        'dir_name': 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8',
        'download_path': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz',
        'height': 1024,
        'width': 1024,
        'net_config': {
            'fixed_width': None,
            'fixed_height': None,
            'input_image': 'serving_default_input_tensor:0',
            'detections': 'StatefulPartitionedCall:12',
            'classes': 'StatefulPartitionedCall:5',
            'scores': 'StatefulPartitionedCall:8',
            'boxes': 'StatefulPartitionedCall:4',
            'start_id': 1,
        }
    },
}

if __name__ == '__main__':
    import pprint
    pprint.PrettyPrinter(1).pprint(models)
    print('Dictionary of pre-trained models configured')
