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
         TestData.File("apples", Path.Combine("Data", "Images"), "apples.jpg"),
         TestData.File("banana", Path.Combine("Data", "Images"), "banana.jpg", "https://github.com/darth-vader-lg/ML-NET/raw/056c60479304a3b5dbdf129c9bc6e853322bb090/test/data/images/banana.jpg"),
         TestData.File("hotdog", Path.Combine("Data", "Images"), "hotdog.jpg", "https://github.com/darth-vader-lg/ML-NET/raw/056c60479304a3b5dbdf129c9bc6e853322bb090/test/data/images/hotdog.jpg")
      };
      /// <summary>
      /// Image folders
      /// </summary>
      public static TestData[] ImageFolders { get; } = new[]
      {
         TestData.Folder("Carps train", Path.Combine("Data", "Images", "Carps"), "train"),
         TestData.Folder("Carps eval", Path.Combine("Data", "Images", "Carps"), "eval"),
      };
      /// <summary>
      /// Pretrained models
      /// </summary>
      public static TestData[] Models { get; } = new[]
      {
         TestData.File(
            "TF SSD MobileNet v2",
            Path.Combine("Data", "Models", "ObjectDetection"),
            Path.Combine("ssd_mobilenet_v2_320x320_coco17_tpu-8", "saved_model", "saved_model.pb"),
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"),
         TestData.File(
            "Onnx SSD MobileNet v1",
            Path.Combine("Data", "Models", "ObjectDetection"),
            "ssd_mobilenet_v1_10.onnx",
            "https://github.com/onnx/models/raw/2c4732abf3bb4890faed986b21853f7034f9979d/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"),
         TestData.File(
            "Onnx YoloV5s",
            Path.Combine("Data", "Models", "ObjectDetection"),
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
