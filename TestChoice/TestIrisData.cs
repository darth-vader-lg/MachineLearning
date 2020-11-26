namespace TestChoice
{
   internal static class TestIrisData
   {
      internal static readonly IrisData Setosa = new IrisData
      {
         SepalLength = 5.1f,
         SepalWidth = 3.5f,
         PetalLength = 1.4f,
         PetalWidth = 0.2f
      };
      internal static readonly IrisData Versicolor = new IrisData
      {
         SepalLength = 7.0f,
         SepalWidth = 3.2f,
         PetalLength = 4.7f,
         PetalWidth = 1.4f
      };
      internal static readonly IrisData Virginica = new IrisData
      {
         SepalLength = 6.3f,
         SepalWidth = 3.3f,
         PetalLength = 6.0f,
         PetalWidth = 2.5f
      };
   }
}
