using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   /// <summary>
   /// Risultato del test algoritmo di classificazione immagini
   /// </summary>
   public class PageImageClassificationPrediction
   {
      /// <summary>
      /// Nome della previsione
      /// </summary>
      [ColumnName("PredictedLabel")]
      public string Prediction { get; set; }
      /// <summary>
      /// Punteggio
      /// </summary>
      public float[] Score { get; set; }
   }
}
