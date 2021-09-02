using MachineLearning.Model;
using MachineLearning.Serialization;
using ObjectDetection.Tests;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("ObjectDetection")]
   public class ObjectDetectionSerialization : TestEnv
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public ObjectDetectionSerialization(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Generic model convertion to ML.NET test
      /// </summary>
      /// <param name="modelPath">Path of the model to import and convert</param>
      [Fact, Trait("Category", "Serialization")]
      public void Clone()
      {
         // Image file
         var imagePath = GetImage("banana").Path;
         Assert.NotNull(imagePath);
         // Import the model
         var model = new ObjectDetection
         {
            ModelStorage = new ModelStorageMemory() { ImportPath = GetModel("Onnx SSD MobileNet v1").Path }
         };
         // Do a prediction
         var prediction = model.GetPrediction(imagePath);
         // Ensure the model is settled and all background task are completed
         model.StopTraining();
         // Clone the model
         var cloned = Serializer.Clone(model);
         model.Dispose();
         // Do a prediction
         var clonedPrediction = cloned.GetPrediction(imagePath);
         // Compare predictions
         foreach (var predition in prediction.GetBoxes().Zip(clonedPrediction.GetBoxes())) {
            // Name, identifier and score of the predictions
            Assert.Equal(predition.First.Name, predition.Second.Name);
            Assert.Equal(predition.First.Id, predition.Second.Id);
            Assert.Equal(predition.First.Score, predition.Second.Score, 3);
            // Detection box size and position
            Assert.Equal(predition.First.Height, predition.Second.Height, 1);
            Assert.Equal(predition.First.Left, predition.Second.Left, 1);
            Assert.Equal(predition.First.Top, predition.Second.Top, 1);
            Assert.Equal(predition.First.Width, predition.Second.Width, 1);
         };
      }
      #endregion
   }
}
