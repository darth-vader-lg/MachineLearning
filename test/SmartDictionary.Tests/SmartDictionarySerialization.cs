using Common.Tests;
using MachineLearning.Serialization;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("SmartDictionary")]
   public class SmartDictionarySerialization : BaseEnvironment
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public SmartDictionarySerialization(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Test smart dictionary cloned with binary serialization
      /// </summary>
      [Fact, Trait("Category", "Serialization")]
      public void Clone()
      {
         // Create the dictionary
         var dictionary = new SmartDictionary<int>()
         {
            { "bottle", 10 },
            { "glass", 5 },
            { "dish", 8 }
         };
         // Tests for key similarities
         var tests = new[]
         {
            (Key: "Bottles in the fridge", Value: 10),
            (Key: "Glasses on the table", Value: 5),
            (Key: "Dishes in the dish drainer", Value: 8)
         };
         // Collect results
         var results = tests.Select(test => (Test: test, Value: dictionary.Similar[test.Key])).ToArray();
         // Ensure the model is settled and all background task are completed
         dictionary.StopTraining();
         // Clone the dictionary
         var cloned = Serializer.Clone(dictionary);
         // Collect results
         var clonedResults = tests.Select(test => (Test: test, Value: cloned.Similar[test.Key])).ToArray();
         // Checks
         foreach (var test in results.Zip(clonedResults)) {
            Assert.Equal(test.First.Test.Value, test.First.Value);
            Assert.Equal(test.Second.Test.Value, test.Second.Value);
            Assert.Equal(test.First.Value, test.Second.Value);
         }
      } 
      #endregion
   }
}
