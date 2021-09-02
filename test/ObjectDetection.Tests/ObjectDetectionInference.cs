using MachineLearning.Model;
using ObjectDetection.Tests;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ObjectDetection")]
   public class ObjectDetectionInference : TestEnv
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ObjectDetectionInference(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Import and inference with Onnx model
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void Onnx() => InferenceSingleThread("Onnx SSD MobileNet v1", 0.7, "banana", "hotdog", "apples");
      /// <summary>
      /// Import and inference with Onnx model and multithread predictions
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void OnnxMultiThread() => InferenceMultiThread("Onnx SSD MobileNet v1", 0.7, "banana", "hotdog", "apples");
      /// <summary>
      /// Import and inference with TensorFlow saved_model
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void TensorFlow() => InferenceSingleThread("TF SSD MobileNet v2", 0.7, "banana", "hotdog", "apples");
      /// <summary>
      /// Import and inference with TensorFlow saved_model and multithread predictions
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void TensorFlowMultiThread() => InferenceMultiThread("TF SSD MobileNet v2", 0.7, "banana", "hotdog", "apples");
      /// <summary>
      /// Import and inference with Onnx YoloV3 model
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void YoloV5s() => InferenceSingleThread("Onnx YoloV5s", 0.5, "banana", "hotdog", "apples");
      /// <summary>
      /// Import and inference with Onnx YoloV3 model
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void YoloV5sMultiThread() => InferenceMultiThread("Onnx YoloV5s", 0.5, "banana", "hotdog", "apples");
      /// <summary>
      /// Generic inference model test
      /// </summary>
      /// <param name="model">Name of the model</param>
      /// <param name="confidence">The minimum prediction confidence required</param>
      /// <param name="images">Set of names of the images to infer</param>
      private void InferenceSingleThread(string model, double confidence, params string[] images)
      {
         // Import the model
         var m = new ObjectDetection
         {
            ModelStorage = new ModelStorageMemory { ImportPath = GetModel(model).Path }
         };
         // Do and check predictions
         for (var i = 0; i < 50; i++) {
            foreach (var image in images) {
               var testImage = GetImage(image);
               var prediction = m.GetPrediction(testImage.Path);
               var boxes = prediction.GetBoxes(confidence);
               Assert.True(boxes.Count > 0, $"No detection boxes found for the image {testImage.Name}");
               if (i == 0) {
                  var best = boxes.OrderBy(box => box.Score).Last();
                  WriteLine($"{testImage.Name}: class {best.Id}, score {best.Score*100:###.#}%");
               }
            }
         }
      }
      /// <summary>
      /// Generic inference test with multithread predictions
      /// </summary>
      /// <param name="model">Name of the model</param>
      /// <param name="confidence">The minimum prediction confidence required</param>
      /// <param name="images">Set of names of the images to infer</param>
      private void InferenceMultiThread(string model, double confidence, params string[] images)
      {
         // Import the model
         var m = new ObjectDetection
         {
            ModelStorage = new ModelStorageMemory { ImportPath = GetModel(model).Path }
         };
         // Do the first prediction to wait the model ready
         m.GetPrediction(GetImage(images[0]).Path);
         // Do and check predictions
         Parallel.For(0, 50, i =>
         {
            foreach (var image in images) {
               var testImage = GetImage(image);
               var prediction = m.GetPrediction(testImage.Path);
               var boxes = prediction.GetBoxes(confidence);
               Assert.True(boxes.Count > 0, $"No detection boxes found for the image {testImage.Name}");
               if (i == 0) {
                  var best = boxes.OrderBy(box => box.Score).Last();
                  WriteLine($"{testImage.Name}: class {best.Id}, score {best.Score * 100:###.#}%");
               }
            }
         });
      }
      #endregion
   }
}
