using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace ML.Utilities
{
   /// <summary>
   /// Estensioni della categoria di classificazione binaria
   /// </summary>
   public static class BinaryClassificationExtensions
   {
      /// <summary>
      /// Estrae il migliore da un elenco di risultati
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il risultato migliore</returns>
      public static TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics> Best(this IEnumerable<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> results)
      {
         try {
            var result = (from item in results
                          orderby item.Metrics.Accuracy descending
                          select item).First();
            return result;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return null;
         }
      }
      /// <summary>
      /// Estrae il migliore da un elenco di risultati
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il risultato migliore</returns>
      public static TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics> Best(this IEnumerable<TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics>> results)
      {
         try {
            var result = (from item in results
                          orderby item.Metrics.Accuracy descending
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
      public static string ToText(this IEnumerable<BinaryClassificationMetrics> metrics)
      {
         try {
            var sb = new StringBuilder();
            var accuracy = metrics.Select(m => m.Accuracy).Where(m => double.IsFinite(m));
            var areaUnderPrecisionRecallCurve = metrics.Select(m => m.AreaUnderPrecisionRecallCurve).Where(m => double.IsFinite(m));
            var areaUnderRocCurve = metrics.Select(m => m.AreaUnderRocCurve).Where(m => double.IsFinite(m));
            var F1Score = metrics.Select(m => m.F1Score).Where(m => double.IsFinite(m));
            var negativePrecision = metrics.Select(m => m.NegativePrecision).Where(m => double.IsFinite(m));
            var negativeRecall = metrics.Select(m => m.NegativeRecall).Where(m => double.IsFinite(m));
            var positivePrecision = metrics.Select(m => m.PositivePrecision).Where(m => double.IsFinite(m));
            var positiveRecall = metrics.Select(m => m.PositiveRecall).Where(m => double.IsFinite(m));
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Average Accuracy:                      {accuracy.Average():0.###} ");
            sb.AppendLine($"*       Average AreaUnderPrecisionRecallCurve: {areaUnderPrecisionRecallCurve.Average():0.###}  ");
            sb.AppendLine($"*       Average AreaUnderRocCurve:             {areaUnderRocCurve.Average():0.###}  ");
            sb.AppendLine($"*       Average F1Score:                       {F1Score.Average():0.###}  ");
            sb.AppendLine($"*       Average NegativePrecision:             {negativePrecision.Average():0.###}  ");
            sb.AppendLine($"*       Average NegativeRecall:                {negativeRecall.Average():0.###}  ");
            sb.AppendLine($"*       Average PositivePrecision:             {positivePrecision.Average():0.###}  ");
            sb.AppendLine($"*       Average PositiveRecall:                {positiveRecall.Average():0.###}  ");
            sb.Append($"*************************************************************************************************************");
            return sb.ToString();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return metrics.ToString();
         }
      }
      /// <summary>
      /// Conversione a testo di una media di metriche
      /// </summary>
      /// <param name="metrics">Elenco di metriche</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<CalibratedBinaryClassificationMetrics> metrics)
      {
         try {
            var sb = new StringBuilder();
            var accuracy = metrics.Select(m => m.Accuracy).Where(m => double.IsFinite(m));
            var areaUnderPrecisionRecallCurve = metrics.Select(m => m.AreaUnderPrecisionRecallCurve).Where(m => double.IsFinite(m));
            var areaUnderRocCurve = metrics.Select(m => m.AreaUnderRocCurve).Where(m => double.IsFinite(m));
            var entropy = metrics.Select(m => m.Entropy).Where(m => double.IsFinite(m));
            var F1Score = metrics.Select(m => m.F1Score).Where(m => double.IsFinite(m));
            var logLoss = metrics.Select(m => m.LogLoss).Where(m => double.IsFinite(m));
            var logLossReduction = metrics.Select(m => m.LogLossReduction).Where(m => double.IsFinite(m));
            var negativePrecision = metrics.Select(m => m.NegativePrecision).Where(m => double.IsFinite(m));
            var negativeRecall = metrics.Select(m => m.NegativeRecall).Where(m => double.IsFinite(m));
            var positivePrecision = metrics.Select(m => m.PositivePrecision).Where(m => double.IsFinite(m));
            var positiveRecall = metrics.Select(m => m.PositiveRecall).Where(m => double.IsFinite(m));
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Average Accuracy:                      {accuracy.Average():0.###} ");
            sb.AppendLine($"*       Average AreaUnderPrecisionRecallCurve: {areaUnderPrecisionRecallCurve.Average():0.###}  ");
            sb.AppendLine($"*       Average AreaUnderRocCurve:             {areaUnderRocCurve.Average():0.###}  ");
            sb.AppendLine($"*       Average Entropy:                       {entropy.Average():0.###}  ");
            sb.AppendLine($"*       Average F1Score:                       {F1Score.Average():0.###}  ");
            sb.AppendLine($"*       Average LogLoss:                       {logLoss.Average():0.###}  ");
            sb.AppendLine($"*       Average LogLossReduction:              {logLossReduction.Average():0.###}  ");
            sb.AppendLine($"*       Average NegativePrecision:             {negativePrecision.Average():0.###}  ");
            sb.AppendLine($"*       Average NegativeRecall:                {negativeRecall.Average():0.###}  ");
            sb.AppendLine($"*       Average PositivePrecision:             {positivePrecision.Average():0.###}  ");
            sb.AppendLine($"*       Average PositiveRecall:                {positiveRecall.Average():0.###}  ");
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
      public static string ToText(this BinaryClassificationMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Accuracy:                     {metrics.Accuracy:0.###} ");
            sb.AppendLine($"*       AreaUnderPrecisionRecallCurve:{metrics.AreaUnderPrecisionRecallCurve:0.###}  ");
            sb.AppendLine($"*       AreaUnderRocCurve:            {metrics.AreaUnderRocCurve:0.###}  ");
            sb.AppendLine($"*       F1Score:                      {metrics.F1Score:0.###}  ");
            sb.AppendLine($"*       NegativePrecision:            {metrics.NegativePrecision:0.###}  ");
            sb.AppendLine($"*       NegativeRecall:               {metrics.NegativeRecall:0.###}  ");
            sb.AppendLine($"*       PositivePrecision:            {metrics.PositivePrecision:0.###}  ");
            sb.AppendLine($"*       PositiveRecall:               {metrics.PositiveRecall:0.###}  ");
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
      public static string ToText(this CalibratedBinaryClassificationMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Accuracy:                     {metrics.Accuracy:0.###} ");
            sb.AppendLine($"*       AreaUnderPrecisionRecallCurve:{metrics.AreaUnderPrecisionRecallCurve:0.###}  ");
            sb.AppendLine($"*       AreaUnderRocCurve:            {metrics.AreaUnderRocCurve:0.###}  ");
            sb.AppendLine($"*       Entropy:                      {metrics.Entropy:0.###}  ");
            sb.AppendLine($"*       F1Score:                      {metrics.F1Score:0.###}  ");
            sb.AppendLine($"*       LogLoss:                      {metrics.LogLoss:0.###}  ");
            sb.AppendLine($"*       LogLossReduction:             {metrics.LogLossReduction:0.###}  ");
            sb.AppendLine($"*       NegativePrecision:            {metrics.NegativePrecision:0.###}  ");
            sb.AppendLine($"*       NegativeRecall:               {metrics.NegativeRecall:0.###}  ");
            sb.AppendLine($"*       PositivePrecision:            {metrics.PositivePrecision:0.###}  ");
            sb.AppendLine($"*       PositiveRecall:               {metrics.PositiveRecall:0.###}  ");
            sb.Append($"*************************************************************************************************************");
            return sb.ToString();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return metrics.ToString();
         }
      }
      /// <summary>
      /// Conversione a testo di una validazione incrociata
      /// </summary>
      /// <param name="result">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics> result) => ToText(result.Metrics);
      /// <summary>
      /// Conversione a testo di una validazione incrociata
      /// </summary>
      /// <param name="result">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics> result) => ToText(result.Metrics);
      /// <summary>
      /// Conversione a testo di una serie di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> results) => ToText(from result in results select result.Metrics);
      /// <summary>
      /// Conversione a testo di una serie di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics>> results) => ToText(from result in results select result.Metrics);
   }
}
