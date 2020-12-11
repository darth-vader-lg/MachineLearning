using Microsoft.ML.Data;
using System;
using System.Diagnostics;
using System.Text;

namespace ML.Utilities
{
   /// <summary>
   /// Estensioni della categoria rilevazioni anomalie
   /// </summary>
   public static class AnomalyDetectionExtensions
   {
      /// <summary>
      /// Conversione a testo della metrica
      /// </summary>
      /// <param name="metrics">Metrica</param>
      /// <returns>Il testo</returns>
      public static string ToText(this AnomalyDetectionMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       AreaUnderRocCurve:                  {metrics.AreaUnderRocCurve:0.###} ");
            sb.AppendLine($"*       DetectionRateAtFalsePositiveCount:  {metrics.DetectionRateAtFalsePositiveCount:0.###}  ");
            sb.AppendLine($"*************************************************************************************************************");
            return sb.ToString();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return metrics.ToString();
         }
      }
   }
}
