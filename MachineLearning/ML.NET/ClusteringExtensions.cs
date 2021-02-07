using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Estensioni della categoria di clusterizzazione
   /// </summary>
   public static class ClusteringExtensions
   {
      /// <summary>
      /// Estrae il migliore da un elenco di modelli e metriche
      /// </summary>
      /// <param name="models">Elenco di modelli e metriche</param>
      /// <returns>Il risultato migliore</returns>
      public static (ITransformer Model, ClusteringMetrics Metrics) Best(this IEnumerable<(ITransformer Model, ClusteringMetrics Metrics)> models)
      {
         try {
            var result = (from item in models
                          orderby item.Metrics.AverageDistance
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
      public static string ToText(this IEnumerable<ClusteringMetrics> metrics)
      {
         try {
            var sb = new StringBuilder();
            var averageDistance = metrics.Select(m => m.AverageDistance).Where(m => double.IsFinite(m));
            var daviesBouldinIndex = metrics.Select(m => m.DaviesBouldinIndex).Where(m => double.IsFinite(m));
            var normalizedMutualInformation = metrics.Select(m => m.NormalizedMutualInformation).Where(m => double.IsFinite(m));
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Average Distance:                   {averageDistance.Average():0.###} ");
            sb.AppendLine($"*       Average DaviesBouldinIndex:         {daviesBouldinIndex.Average():0.###}  ");
            sb.AppendLine($"*       Average NormalizedMutualInformation:{normalizedMutualInformation.Average():0.###}  ");
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
      public static string ToText(this ClusteringMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       AverageDistance:              {metrics.AverageDistance:0.###} ");
            sb.AppendLine($"*       DaviesBouldinIndex:           {metrics.DaviesBouldinIndex:0.###}  ");
            sb.AppendLine($"*       NormalizedMutualInformation:  {metrics.NormalizedMutualInformation:0.###}  ");
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
      public static string ToText(this TrainCatalogBase.CrossValidationResult<ClusteringMetrics> result) => ToText(result.Metrics);
      /// <summary>
      /// Conversione a testo di una serie di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il testo</returns>
      public static string ToText(this IEnumerable<TrainCatalogBase.CrossValidationResult<ClusteringMetrics>> results) => ToText(from result in results select result.Metrics);
   }
}
