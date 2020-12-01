using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   /// <summary>
   /// Dati di test algoritmo K-Means per la clusterizzazione dei piedi
   /// </summary>
   public class PageFeetKMeansData
   {
      [LoadColumn(0)]
      public string Number;
      [LoadColumn(1)]
      public float Length;
      [LoadColumn(2)]
      public float Instep;
   }
}
