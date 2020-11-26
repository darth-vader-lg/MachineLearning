using Microsoft.ML.Data;

namespace TestChoice
{
   public class IrisPrediction
   {
      [ColumnName("PredictedLabel")]
      public uint PredictedClusterId;
      [ColumnName("Score")]
      public float[] Distances;
   }
}
