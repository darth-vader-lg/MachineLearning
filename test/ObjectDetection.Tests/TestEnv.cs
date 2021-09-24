using Common.Tests;
using System;
using System.IO;
using System.Linq;
using Xunit.Abstractions;

namespace ObjectDetection.Tests
{
   /// <summary>
   /// Environment for the object detection tests
   /// </summary>
   public class TestEnv : BaseEnvironment
   {
      #region Properties
      /// <summary>
      /// Test images
      /// </summary>
      public static TestData[] Images { get; } = new[]
      {
         TestData.File("apples", Path.Combine(DataFolder, "Images"), "apples.jpg"),
         TestData.File("banana", Path.Combine(DataFolder, "Images"), "banana.jpg", "https://github.com/darth-vader-lg/ML-NET/raw/056c60479304a3b5dbdf129c9bc6e853322bb090/test/data/images/banana.jpg"),
         TestData.File("hotdog", Path.Combine(DataFolder, "Images"), "hotdog.jpg", "https://github.com/darth-vader-lg/ML-NET/raw/056c60479304a3b5dbdf129c9bc6e853322bb090/test/data/images/hotdog.jpg")
      };
      /// <summary>
      /// Image folders
      /// </summary>
      public static TestData[] ImageFolders { get; } = new[]
      {
         TestData.Folder("Carps train", Path.Combine(DataFolder, "Images", "Carps"), "train"),
         TestData.Folder("Carps eval", Path.Combine(DataFolder, "Images", "Carps"), "eval"),
      };
      /// <summary>
      /// Pretrained models
      /// </summary>
      public static TestData[] Models { get; } = new[]
      {
         // TensorFlow2 Model Zoo
         TestData.File(
            "TF CenterNet HourGlass104 512x512",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("centernet_hg104_512x512_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF CenterNet HourGlass104 1024x1024",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("centernet_hg104_1024x1024_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF CenterNet Resnet50 V1 FPN 512x512",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("centernet_resnet50_v1_fpn_512x512_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF CenterNet Resnet101 V1 FPN 512x512",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("centernet_resnet101_v1_fpn_512x512_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF CenterNet Resnet50 V2 512x512",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("centernet_resnet50_v2_512x512_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF CenterNet MobileNet V2",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("centernet_mobilenetv2_fpn_od", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"),
         TestData.File(
            "TF Efficientdet D0 512x512",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d0_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF Efficientdet D1 640x640",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d1_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF Efficientdet D2 768x768",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d2_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF Efficientdet D3 896x896",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d3_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF Efficientdet D4 1024x1024",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d4_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF Efficientdet D5 1280x1280",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d5_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF Efficientdet D6 1280x1280", // Really speaking it's 1408
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d6_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF Efficientdet D7 1536x1536",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("efficientdet_d7_coco17_tpu-32", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz"),
         TestData.File(
            "TF SSD MobileNet V2 320x320",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_mobilenet_v2_320x320_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD MobileNet V2 FPNLite 320x320", //Really speaking it's 300
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD MobileNet V1 FPN 640x640",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD MobileNet V2 FPNLite 640x640",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD ResNet50 V1 FPN 640x640 (RetinaNet50)",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD ResNet101 V1 FPN 640x640 (RetinaNet101)",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_resnet101_v1_fpn_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD ResNet152 V1 FPN 640x640 (RetinaNet152)",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_resnet152_v1_fpn_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF SSD Faster R-CNN ResNet50 V1 640x640",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet50_v1_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet50 V1 1024x1024",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet50 V1 800x1333",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet101 V1 640x640",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet101_v1_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet101 V1 1024x1024",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet101 V1 800x1333",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet152 V1 640x640",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet101_v1_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet152 V1 1024x1024",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN ResNet152 V1 800x1333",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN Inception ResNet V2 640x640",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Faster R-CNN Inception ResNet V2 1024x1024", // Really speaking it's 800x1333
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz"),
         TestData.File(
            "TF Mask R-CNN Inception ResNet V2 1024x1024",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            Path.Combine("mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"),
         // Onnx
         TestData.File(
            "Onnx SSD MobileNet v1",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            "ssd_mobilenet_v1_10.onnx",
            "https://github.com/onnx/models/raw/2c4732abf3bb4890faed986b21853f7034f9979d/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"),
         TestData.File(
            "Onnx YoloV5s",
            Path.Combine(DataFolder, "Models", "ObjectDetection"),
            "yolov5s.onnx",
            "https://github.com/nihevix/yolov5-net-master/raw/6d8d690f04342f9fb38b7b4390338968e702bdf1/src/Yolov5Net.App/Assets/Weights/yolov5s.onnx"),
      };
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      internal TestEnv(ITestOutputHelper output = null) : base(output)
      {
         // Disable cuda for tests
         Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "-1");
      }
      /// <summary>
      /// Return a known model
      /// </summary>
      /// <param name="name">Name of the model</param>
      /// <returns>The model or null</returns>
      public static TestData GetModel(string name) => Models.FirstOrDefault(model => model.Name == name);
      /// <summary>
      /// Return a known image
      /// </summary>
      /// <param name="name">Name of the image</param>
      /// <returns>The path of the image or null</returns>
      public static TestData GetImage(string name) => Images.FirstOrDefault(image => image.Name == name);
      /// <summary>
      /// Return a known images folder
      /// </summary>
      /// <param name="name">Name of the images folder</param>
      /// <returns>The path of the images folder or null</returns>
      public static TestData GetImagesFolder(string name) => ImageFolders.FirstOrDefault(folder => folder.Name == name);
      #endregion
   }
}
