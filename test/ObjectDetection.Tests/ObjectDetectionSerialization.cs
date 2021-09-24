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
         // Original and cloned objects
         var orgModel = default(ObjectDetection);
         var clonedModel = default(ObjectDetection);
         try {
            // Image file
            var imagePath = GetImage("banana").Get();
            Assert.NotNull(imagePath);
            // Import the model
            orgModel = new ObjectDetection { ModelStorage = new ModelStorageMemory() { ImportPath = GetModel("Onnx SSD MobileNet v1").Get() } };
            // Do the predictions
            var orgPrediction = orgModel.GetPrediction(imagePath);
            // Ensure the model is settled and all background task are completed
            orgModel.StopTraining();
            // Clone the model
            clonedModel = Serializer.Clone(orgModel);
            DisposeAndNullify(ref orgModel);
            // Do the predictions
            var clonedPrediction = clonedModel.GetPrediction(imagePath);
            // Compare predictions
            foreach (var (First, Second) in orgPrediction.GetBoxes().Zip(clonedPrediction.GetBoxes())) {
               // Name, identifier and score of the predictions
               Assert.Equal(First.Name, Second.Name);
               Assert.Equal(First.Id, Second.Id);
               Assert.Equal(First.Score, Second.Score, 3);
               // Detection box size and position
               Assert.Equal(First.Height, Second.Height, 1);
               Assert.Equal(First.Left, Second.Left, 1);
               Assert.Equal(First.Top, Second.Top, 1);
               Assert.Equal(First.Width, Second.Width, 1);
            }
         }
         finally {
            DisposeAndNullify(ref orgModel);
            DisposeAndNullify(ref clonedModel);
         };
      }
      #endregion
   }
}
