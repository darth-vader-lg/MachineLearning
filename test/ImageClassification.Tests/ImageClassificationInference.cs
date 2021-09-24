using ImageClassification.Tests;
using MachineLearning.Model;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ImageClassification")]
   public class ImageClassificationInference : TestEnv
   {
      #region class TFHubSet
      /// <summary>
      /// TensorFlow hub models
      /// </summary>
      internal class TFHubSet : IEnumerable<object[]>
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
      public ImageClassificationInference(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test for image classification inference with pretrained tensorflow inception
      /// </summary>
      [Theory, ClassData(typeof(TFHubSet)), Trait("Category", "Inference")]
      public void TensorFlowHub(bool multithread, string model) =>
         Inference(model, 0.6, new[] { "banana", "hotdog" }, new[] { "banana", "hotdog" }, multithread);
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
         using var m = new ImageClassification { ModelStorage = new ModelStorageMemory { ImportPath = GetModel(model).Get() } };
         // Do predictions
         var predictions = new List<ImageClassification.Prediction>();
         void Execute(string image, string label)
         {
            var testImage = GetImage(image);
            var prediction = m.GetPrediction(testImage.Get());
            if (prediction.Kind == label) {
               WriteLine($"{prediction.Kind}: class {prediction.Id}, score {prediction.Score * 100:###.#}%");
               lock (predictions)
                  predictions.Add(prediction);
            }
            else
               WriteLine($"Wrong prediction: expected {label}, got {prediction.Kind}");
         }
         if (multithread)
            Parallel.ForEach(images.Zip(expectedLabels).Select(item => (Image: item.First, Label: item.Second)), item => Execute(item.Image, item.Label));
         else
            images.Zip(expectedLabels).Select(item => (Image: item.First, Label: item.Second)).ToList().ForEach(item => Execute(item.Image, item.Label));
         // Test predictions
         Assert.True(predictions.Count == images.Length, $"Not all images where found in the inference");
         Assert.True(predictions.Count == images.Length, $"Not all images where found in the inference");
         var ap = predictions.Average(p => p.Score);
         WriteLine($"Average precision is {ap * 100:###.#}%");
         Assert.True(ap >= confidence, $"The minimum required precision was {confidence * 100:###.#}%");
      }
      #endregion
   }
}
