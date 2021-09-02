using Common.Tests;
using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Serialization;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("TextMeaning")]
   public class SentenceClassificationSerialization : BaseEnvironment
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public SentenceClassificationSerialization(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test the sentence classification model cloning it with binary serialization
      /// </summary>
      [Fact, Trait("Category", "Serialization")]
      public void Clone()
      {
         // Prepare data
         var data = new StringBuilder();
         data.AppendLine("Open window, open the window");
         data.AppendLine("Close window, close the window");
         data.AppendLine("Open car, open the car");
         data.AppendLine("Close car, close the car");
         data.AppendLine("Switch on light, switch on the light");
         data.AppendLine("Switch off light, switch off the light");
         // Create the model
         var model = new SentenceClassification()
         {
            DataStorage = new DataStorageTextMemory() { TextData = data.ToString() },
            ModelStorage = new ModelStorageMemory(),
            ModelTrainer = new ModelTrainerStandard(),
         };
         // Do prediction
         var prediction = model.GetPrediction(default, "please, open my car");
         Assert.Equal("Open car", prediction.Meaning);
         // Ensure the model is settled and all background task are completed
         model.StopTraining();
         // Clone the model
         var cloned = Serializer.Clone(model);
         // Do prediction
         var clonedPrediction = cloned.GetPrediction(default, "please, could you open my car");
         // Compare results
         Assert.Equal(clonedPrediction.Meaning, prediction.Meaning);
      } 
      #endregion
   }
}
