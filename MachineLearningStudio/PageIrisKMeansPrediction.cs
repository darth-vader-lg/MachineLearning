using Microsoft.ML.Data;

namespace TestChoice
{
   public class PageIrisKMeansPrediction
   {
      [ColumnName("PredictedLabel")]
      public uint PredictedClusterId;
      [ColumnName("Score")]
      public float[] Distances;
   }
}
