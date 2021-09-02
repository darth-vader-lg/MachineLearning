using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Estensioni della categoria di classificazione multiclasse
   /// </summary>
   public static class MulticlassClassificationExtensions
   {
      /// <summary>
      /// Estrae il migliore da un elenco di modelli e metriche
      /// </summary>
      /// <param name="models">Elenco di modelli e metriche</param>
      /// <returns>Il risultato migliore</returns>
      public static (ITransformer Model, MulticlassClassificationMetrics Metrics) Best(this IEnumerable<(ITransformer Model, MulticlassClassificationMetrics Metrics)> models)
      {
         try {
            var result = (from item in (from item in models
                                        group item by item.Metrics.MicroAccuracy into grps
                                        orderby grps.Key descending
                                        select grps).First()
                          orderby item.Metrics.LogLoss
                          select item).First();
            return result;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return default;
         }
      }
      /// <summary>
      /// Conversione a testo di una media di metriche
      /// </summary>
      /// <param name="metrics">Elenco di metriche</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<MulticlassClassificationMetrics> metrics)
      {
         try {
            var sb = new StringBuilder();
            var microAccuracyValues = metrics.Select(m => m.MicroAccuracy).Where(m => double.IsFinite(m));
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = microAccuracyValues.CalculateStandardDeviation();
            var microAccuraciesConfidenceInterval95 = microAccuracyValues.CalculateConfidenceInterval95();
            var macroAccuracyValues = metrics.Select(m => m.MacroAccuracy).Where(m => double.IsFinite(m));
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = macroAccuracyValues.CalculateStandardDeviation();
            var macroAccuraciesConfidenceInterval95 = macroAccuracyValues.CalculateConfidenceInterval95();
            var logLossValues = metrics.Select(m => m.LogLoss).Where(m => double.IsFinite(m));
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = logLossValues.CalculateStandardDeviation();
            var logLossConfidenceInterval95 = logLossValues.CalculateConfidenceInterval95();
            var logLossReductionValues = metrics.Select(m => m.LogLossReduction).Where(m => double.IsFinite(m));
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = logLossReductionValues.CalculateStandardDeviation();
            var logLossReductionConfidenceInterval95 = logLossReductionValues.CalculateConfidenceInterval95();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:0.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:0.###})");
            sb.AppendLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:0.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:0.###})");
            sb.AppendLine($"*       Average LogLoss:          {logLossAverage:0.###}  - Standard deviation: ({logLossStdDeviation:0.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:0.###})");
            sb.AppendLine($"*       Average LogLossReduction: {logLossReductionAverage:0.###}  - Standard deviation: ({logLossReductionStdDeviation:0.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:0.###})");
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
      public static string ToText(this MulticlassClassificationMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            sb.AppendLine($"*       MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            sb.AppendLine($"*       LogLoss:          {metrics.LogLoss:0.###}");
            sb.AppendLine($"*       LogLossReduction: {metrics.LogLossReduction:0.###}");
            sb.AppendLine($"*************************************************************************************************************");
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
      public static string ToText(this TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics> result) => ToText(result.Metrics);
      /// <summary>
      /// Conversione a testo di una serie di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> results) => ToText(from result in results select result.Metrics);
   }
}
