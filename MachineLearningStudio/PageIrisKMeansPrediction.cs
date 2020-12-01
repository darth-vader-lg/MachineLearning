using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   public class PageIrisKMeansPrediction
   {
      [ColumnName("PredictedLabel")]
      public uint PredictedClusterId;
      [ColumnName("Score")]
      public float[] Distances;
   }
}
