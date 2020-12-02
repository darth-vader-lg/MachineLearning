using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   /// <summary>
   /// Dati di test algoritmo di classificazione immagini
   /// </summary>
   public class PageImageClassificationData
   {
      /// <summary>
      /// Classe dell'immagine
      /// </summary>
      [ColumnName("Label"), LoadColumn(0)]
      public string Label { get; set; }
      /// <summary>
      /// Path dell'immagine
      /// </summary>
      [ColumnName("ImageSource"), LoadColumn(1)]
      public string ImageSource { get; set; }
   }
}
