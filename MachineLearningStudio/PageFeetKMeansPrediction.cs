using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   /// <summary>
   /// Dati di test algoritmo K-Means per la clusterizzazione dei piedi
   /// </summary>
   public class PageFeetKMeansPrediction
   {
      [ColumnName("PredictedLabel")]
      public uint PredictedClusterId;
      [ColumnName("Score")]
      public float[] Distances;
   }
}
