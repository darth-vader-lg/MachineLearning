using MachineLearning.Model;
using ObjectDetection.Tests;
using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ObjectDetection")]
   public class ObjectDetectionImport : TestEnv
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ObjectDetectionImport(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Model import from Onnx to ML.NET
      /// </summary>
      /// <param name="modelPath">Path of the model to import</param>
      [Fact, Trait("Category", "ModelImport")]
      public void Onnx() => ImportModel("Onnx SSD MobileNet v1", "banana", "hotdog");
      /// <summary>
      /// Model import from TensorFlow to ML.NET
      /// </summary>
      /// <param name="modelPath">Path of the model to import</param>
      [Fact, Trait("Category", "ModelImport")]
      public void TensorFlow() => ImportModel("TF SSD MobileNet v2", "banana", "hotdog");
      /// <summary>
      /// Generic model import to ML.NET test
      /// </summary>
      /// <param name="model">Name of the model</param>
      /// <param name="images">Set of names of the images to infer</param>
      private void ImportModel(string model, params string[] images)
      {
         // Temporary model path
         var tempModelFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
         try {
            // Import the model
            var m = new ObjectDetection
            {
               ModelStorage = new ModelStorageFile(tempModelFile) { ImportPath = GetModel(model).Path }
            };
            // Do the predictions
            var importedPredictions = (from image in images select m.GetPrediction(GetImage(image).Path)).ToArray();
            m.Dispose();
            // Open the converted model
            m = new ObjectDetection
            {
               ModelStorage = new ModelStorageFile(tempModelFile)
            };
            // Do the predictions
            var convertedPredictions = (from image in images select m.GetPrediction(GetImage(image).Path)).ToArray();
            // Compare predictions
            var importedBoxes = from prediction in importedPredictions from box in prediction.GetBoxes() select box;
            var convertedBoxes = from prediction in convertedPredictions from box in prediction.GetBoxes() select box;
            foreach (var (First, Second) in importedBoxes.Zip(convertedBoxes)) {
               // Name, identifier and score of the predictions
               Assert.Equal(First.Name, Second.Name);
               Assert.Equal(First.Id, Second.Id);
               Assert.Equal(First.Score, Second.Score, 3);
               // Detection box size and position
               Assert.Equal(First.Height, Second.Height, 1);
               Assert.Equal(First.Left, Second.Left, 1);
               Assert.Equal(First.Top, Second.Top, 1);
               Assert.Equal(First.Width, Second.Width, 1);
            };
         }
         finally {
            File.Delete(tempModelFile);
         }
      }
      #endregion
   }
}
