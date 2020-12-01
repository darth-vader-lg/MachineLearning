using Microsoft.ML.Data;

namespace TestChoice
{
   /// <summary>
   /// Dati di test algoritmo di previsione piedi Sdca
   /// </summary>
   public class PageFeetSdcaData
   {
      [LoadColumn(0)]
      public float Number;
      [LoadColumn(1)]
      public float Length;
      [LoadColumn(2)]
      public float Instep;
   }
}
