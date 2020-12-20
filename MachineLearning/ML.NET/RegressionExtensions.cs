using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Estensioni della categoria di regressione
   /// </summary>
   public static class RegressionExtensions
   {
      /// <summary>
      /// Estrae il migliore da un elenco di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il risultato migliore</returns>
      public static TrainCatalogBase.CrossValidationResult<RegressionMetrics> Best(this IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> results)
      {
         try {
            var result = (from item in results
                          orderby item.Metrics.LossFunction
                          select item).First();
            return result;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return null;
         }
      }
      /// <summary>
      /// Conversione a testo di una media di metriche
      /// </summary>
      /// <param name="metrics">Elenco di metriche</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<RegressionMetrics> metrics)
      {
         try {
            var sb = new StringBuilder();
            var L1 = metrics.Select(m => m.MeanAbsoluteError);
            var L2 = metrics.Select(m => m.MeanSquaredError);
            var RMS = metrics.Select(m => m.RootMeanSquaredError);
            var lossFunction = metrics.Select(m => m.LossFunction);
            var R2 = metrics.Select(m => m.RSquared);
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            sb.AppendLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            sb.AppendLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            sb.AppendLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            sb.AppendLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            sb.Append($"*************************************************************************************************************");
            return sb.ToString();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return metrics.ToString();
         }
      }
      /// <summary>
      /// Conversione a testo della metrica
      /// </summary>
      /// <param name="metrics">La metrica</param>
      /// <returns>Il testo</returns>
      public static string ToText(this RegressionMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Regression model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       LossFunction:        {metrics.LossFunction:0.###}");
            sb.AppendLine($"*       MeanAbsoluteError:   {metrics.MeanAbsoluteError:0.###}");
            sb.AppendLine($"*       MeanSquaredError:    {metrics.MeanSquaredError:0.###}");
            sb.AppendLine($"*       RootMeanSquaredError:{metrics.RootMeanSquaredError:0.###}");
            sb.AppendLine($"*       RSquared:            {metrics.RSquared:0.###}");
            sb.Append($"*************************************************************************************************************");
            return sb.ToString();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return metrics.ToString();
         }
      }
      /// <summary>
      /// Conversione a testo di un risultato di validazione incrociata
      /// </summary>
      /// <param name="result">Risultato</param>
      /// <returns>Il testo</returns>
      public static string ToText(this TrainCatalogBase.CrossValidationResult<RegressionMetrics> result) => ToText(result.Metrics);
      /// <summary>
      /// Conversione a testo di una serie di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> results) => ToText(from result in results select result.Metrics);
   }
}
