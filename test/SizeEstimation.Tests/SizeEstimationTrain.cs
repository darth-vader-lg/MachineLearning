using Common.Tests;
using Xunit;
using Xunit.Abstractions;

namespace MachineLearning.ModelZoo.Tests
{
   [Collection("SizeEstimator")]
   public class SizeEstimationTrain : BaseEnvironment
   {
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public SizeEstimationTrain(ITestOutputHelper output) : base(output) { }
      /// <summary>
      /// Train
      /// </summary>
      [Fact, Trait("Category", "Train")]
      public void Train()
      {
      }
      #endregion
   }
}
