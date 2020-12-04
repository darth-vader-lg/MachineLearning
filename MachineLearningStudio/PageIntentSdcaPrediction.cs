using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   /// <summary>
   /// Risultato del test algoritmo di previsione intenzioni
   /// </summary>
   public class PageIntentSdcaPrediction
   {
      /// <summary>
      /// Intenzione prevista
      /// </summary>
      [ColumnName("PredictedLabel")]
      public string Intent { get; set; }
      /// <summary>
      /// Punteggi
      /// </summary>
      public float[] Score { get; set; }
   }
}
