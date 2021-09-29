using MachineLearning.Model;
using ObjectDetection.Tests;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ObjectDetection")]
   public class ObjectDetectionInference : TestEnv
   {
      #region class OnnxSet
      /// <summary>
      /// Onnx models
      /// </summary>
      internal class OnnxSet : IEnumerable<object[]>
      {
         public IEnumerator<object[]> GetEnumerator()
         {
            foreach (var multithread in new[] { false, true }) {
               foreach (var model in Models.Where(m => m.Name.ToLower().Contains("onnx")))
                  yield return new object[] { multithread, model.Name };
            }
         }
         IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
      }
      /// <summary>
      /// PyTorch models
      /// </summary>
      internal class PyTorchSet : IEnumerable<object[]>
      {
         public IEnumerator<object[]> GetEnumerator()
         {
            foreach (var multithread in new[] { false, true }) {
               foreach (var model in Models.Where(m => m.Name.ToLower().Contains("pytorch")))
                  yield return new object[] { multithread, model.Name };
            }
         }
         IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
      }
      #endregion
      #region class TF2ModelZooSet
      /// <summary>
      /// TensorFlow 2 model zoo models
      /// </summary>
      internal class TF2ModelZooSet : IEnumerable<object[]>
      {
         public IEnumerator<object[]> GetEnumerator()
         {
            foreach (var multithread in new[] { false, true }) {
               foreach (var model in Models.Where(m => m.Name.StartsWith("TF ")))
                  yield return new object[] { multithread, model.Name };
            }
         }
         IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
      } 
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ObjectDetectionInference(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Import and inference with Onnx model
      /// </summary>
      [Theory, ClassData(typeof(OnnxSet)), Trait("Category", "Inference")]
      public void Onnx(bool multithread, string model) =>
         Inference(model, 0.5, new[] { "banana", "hotdog", "apples" }, new[] { "banana", "hot dog", "apple" }, multithread);
      /// <summary>
      /// Import and inference with PyTorch models
      /// </summary>
      [Theory, ClassData(typeof(PyTorchSet)), Trait("Category", "Inference")]
      public void YoloV5(bool multithread, string model) =>
         Inference(model, 0.5, new[] { "bus and persons", "bus and persons", "zidane" }, new[] { "bus", "person", "person" }, multithread);
      /// <summary>
      /// Import and inference with TensorFlow2 model zoo (saved_model)
      /// </summary>
      [Theory, ClassData(typeof(TF2ModelZooSet)), Trait("Category", "Inference")]
      public void TensorFlowModelZoo2(bool multithread, string model) =>
         Inference(model, 0.5, new[] { "banana", "hotdog", "apples" }, new[] { "banana", "hot dog", "apple" }, multithread);
      /// <summary>
      /// Generic inference model test
      /// </summary>
      /// <param name="model">Name of the model</param>
      /// <param name="confidence">The minimum prediction confidence required</param>
      /// <param name="images">Set of names of the images to infer</param>
      /// <param name="expectedLabels">Set of expected labels</param>
      /// <param name="multithread">Enable multithread inference</param>
      private void Inference(string model, double confidence, string[] images, string[] expectedLabels, bool multithread = false)
      {
         // Checks
         Assert.NotNull(images);
         Assert.NotNull(expectedLabels);
         Assert.True(images.Length == expectedLabels.Length);
         // Import the model
         using var m = new ObjectDetection { ModelStorage = new ModelStorageMemory { ImportPath = GetModel(model).Get() } };
         // Do predictions
         var dets = new List<ObjectDetection.Prediction.Box>();
         void Execute(string image, string label)
         {
            var testImage = GetImage(image);
            var prediction = m.GetPrediction(testImage.Get());
            var boxes = prediction.GetBoxes();
            var best = boxes.Where(box => box.Name == label).OrderBy(box => box.Score).LastOrDefault();
            if (best != null) {
               WriteLine($"{best.Name}: class {best.Id}, score {best.Score * 100:###.#}%");
               lock (dets)
                  dets.Add(best);
            }
            else
               WriteLine($"No detection boxes found for the image {testImage.Name}");
         }
         if (multithread)
            Parallel.ForEach(images.Zip(expectedLabels).Select(item => (Image: item.First, Label: item.Second)), item => Execute(item.Image, item.Label));
         else
            images.Zip(expectedLabels).Select(item => (Image: item.First, Label: item.Second)).ToList().ForEach(item => Execute(item.Image, item.Label));
         // Test predictions
         Assert.True(dets.Count == images.Length, $"Not all images where found in the inference");
         var ap = dets.Average(det => det.Score);
         WriteLine($"Average precision is {ap * 100:###.#}%");
         Assert.True(ap >= confidence, $"The minimum required precision was {confidence * 100:###.#}%");
      }
      #endregion
   }
}
