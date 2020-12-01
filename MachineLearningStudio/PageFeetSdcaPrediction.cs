using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   /// <summary>
   /// Risultato del test algoritmo di previsione piedi Sdca
   /// </summary>
   public class PageFeetSdcaPrediction
   {
      [ColumnName("Score")]
      public float Number;
   }
}
