using Microsoft.ML.Data;

namespace TestChoice
{
   public class FeetPrediction
   {
      [ColumnName("PredictedLabel")]
      public uint PredictedClusterId;
      [ColumnName("Score")]
      public float[] Distances;
   }
}
