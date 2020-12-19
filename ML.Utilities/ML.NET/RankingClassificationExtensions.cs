using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Estensioni della categoria di ranking
   /// </summary>
   public static class RankingClassificationExtensions
   {
      /// <summary>
      /// Estrae il migliore da un elenco di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il risultato migliore</returns>
      public static TrainCatalogBase.CrossValidationResult<RankingMetrics> Best(this IEnumerable<TrainCatalogBase.CrossValidationResult<RankingMetrics>> results)
      {
         try {
            var result = (from item in results
                          orderby DcgScore(item.Metrics.NormalizedDiscountedCumulativeGains) descending
                          select item).First();
            return result;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            return null;
         }
      }
      /// <summary>
      /// Funzione del calcolo dello score
      /// </summary>
      /// <param name="dcgs">Elenco di discounted cumulative gains</param>
      /// <returns>Lo score</returns>
      internal static double DcgScore(IEnumerable<double> dcgs)
      {
         var i = 2.0;
         var result = 0.0;
         foreach (var dcg in dcgs)
            result += dcg / Math.Log(i++);
         return result;
      }
      /// <summary>
      /// Conversione a testo di una media di metriche
      /// </summary>
      /// <param name="metrics">Elenco di metriche</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<RankingMetrics> metrics)
      {
         try {
            var sb = new StringBuilder();
            var discountedCumulativeGains = metrics.Select(m => DcgScore(m.DiscountedCumulativeGains));
            var normalizedDiscountedCumulativeGains = metrics.Select(m => DcgScore(m.NormalizedDiscountedCumulativeGains));
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Ranking model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       Average DiscountedCumulativeGains:           {discountedCumulativeGains.Average():0.###} ");
            sb.AppendLine($"*       Average NormalizedDiscountedCumulativeGains: {normalizedDiscountedCumulativeGains.Average():0.###} ");
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
      public static string ToText(this RankingMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       DCG Score:           {DcgScore(metrics.DiscountedCumulativeGains)}");
            sb.AppendLine($"*       NDCG Score:          {DcgScore(metrics.NormalizedDiscountedCumulativeGains)}");
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
      public static string ToText(this TrainCatalogBase.CrossValidationResult<RankingMetrics> result) => ToText(result.Metrics);
      /// <summary>
      /// Conversione a testo di una serie di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<TrainCatalogBase.CrossValidationResult<RankingMetrics>> results) => ToText(from result in results select result.Metrics);
   }
}
