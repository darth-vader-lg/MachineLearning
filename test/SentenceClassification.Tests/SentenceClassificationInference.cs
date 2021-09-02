using Common.Tests;
using MachineLearning.Data;
using MachineLearning.Model;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("TextMeaning")]
   public class SentenceClassificationInference : BaseEnvironment
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public SentenceClassificationInference(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test of the text comprehension
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void InferenceSingleThread()
      {
         // Get the model
         var model = GetModel();
         // Get the test set
         var tests = GetTests();
         // Inference loop
         for (int i = 0; i < 4; i++) {
            foreach (var test in GetTests()) {
               var prediction = model.GetPrediction(default, test.Sentence);
               Assert.Equal(test.Expected, prediction.Meaning);
            }
         }
      }
      /// <summary>
      /// Test of the text comprehension in multithread
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void InferenceMultiThread()
      {
         // Get the model
         var model = GetModel();
         // Get the test set
         var tests = GetTests();
         // Wait for prediction availables
         model.GetPrediction(default, tests[0].Sentence);
         // Inference loop
         Parallel.For(0, 4, n =>
         {
            foreach (var test in tests) {
               var prediction = model.GetPrediction(default, test.Sentence);
               Assert.Equal(test.Expected, prediction.Meaning);
            }
         });
      }
      /// <summary>
      /// Prepare the model
      /// </summary>
      /// <returns>The model</returns>
      private SentenceClassification GetModel()
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
         return model;
      }
      /// <summary>
      /// Return a list of sentences and meanings to understand
      /// </summary>
      /// <returns>The list</returns>
      private (string Sentence, string Expected)[] GetTests()
      {
         return new[]
         {
            ("Open the closed windows", "Open window"),
            ("Close every windows", "Close window"),
            ("Please, open my car", "Open car"),
            ("Close the car before going away", "Close car"),
            ("I don't see anything! Switch on the light", "Switch on light"),
            ("I like to see the television with the lights switched off", "Switch off light"),
         };
      }
      #endregion
   }
}
