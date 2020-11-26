using Microsoft.ML.Data;

namespace TestChoice
{
   public class FeetData
   {
      [LoadColumn(0)]
      public string Number;
      [LoadColumn(1)]
      public float Length;
      [LoadColumn(2)]
      public float Instep;
   }
}
