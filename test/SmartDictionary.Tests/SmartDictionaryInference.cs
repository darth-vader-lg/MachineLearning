using Common.Tests;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("SmartDictionary")]
   public class SmartDictionaryInference : BaseEnvironment
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public SmartDictionaryInference(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Check for equality in the key of a dictionary
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void KeyEquality()
      {
         var dictionary = new SmartDictionary<string>()
         {
            { "this is a house", "house" },
            { "this is a car", "car" },
            { "this is a window", "window" },
         };
         var result = dictionary["this is a house"];
         Assert.Equal("house", result);
         Assert.Throws<KeyNotFoundException>(() => dictionary["Is this a house?"]);
      }
      /// <summary>
      /// Check for similarity in the key of a dictionary
      /// </summary>
      [Fact, Trait("Category", "Inference")]
      public void KeySimilarity()
      {
         var dictionary = new SmartDictionary<string>()
         {
            { "this is a house", "house" },
            { "this is a car", "car" },
            { "this is a window", "window" },
         };
         var result = dictionary.Similar["these are houses"];
         Assert.Equal("house", result);
         result = dictionary.Similar["I see a car"];
         Assert.Equal("car", result);
         result = dictionary.Similar["It seems like a window"];
         Assert.Equal("window", result);
      }
      #endregion
   }
}
