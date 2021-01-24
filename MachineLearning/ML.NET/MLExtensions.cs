using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Microsoft.ML
{
   /// <summary>
   /// Estensioni generiche
   /// </summary>
   public static class MLExtensions
   {
      /// <summary>
      /// Calcola l'intervallo di fiducia 95%
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>L'intervallo di fiducia 95%</returns>
      internal static double CalculateConfidenceInterval95(this IEnumerable<double> values)
      {
         var confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
         return confidenceInterval95;
      }
      /// <summary>
      /// Calcola la deviazione standard di un set di valori
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>La deviazione standard</returns>
      internal static double CalculateStandardDeviation(this IEnumerable<double> values)
      {
         var average = values.Average();
         var sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
         var standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
         return standardDeviation;
      }
      /// <summary>
      /// Restituisce un progress per l'autotraining di rilevamento anomalie
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="contextName">Nome del contesto</param>
      /// <param name="reportRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<AnomalyDetectionMetrics> AnomalyDetectionProgress(
         this MLContext ml,
         string contextName = null,
         AutoMLProgress<AnomalyDetectionMetrics>.ReportRunDetails reportRunDetails = null,
         AutoMLProgress<AnomalyDetectionMetrics>.ReportCrossValidationRunDetails reportCrossValidationRunDetails = null) =>
         new AutoMLProgress<AnomalyDetectionMetrics>(ml, contextName, reportRunDetails, reportCrossValidationRunDetails);
      /// <summary>
      /// Restituisce un progress per l'autotraining di classificazione binaria
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="contextName">Nome del contesto</param>
      /// <param name="reportRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<BinaryClassificationMetrics> BinaryClassificationProgress(
         this MLContext ml,
         string contextName = null,
         AutoMLProgress<BinaryClassificationMetrics>.ReportRunDetails reportRunDetails = null,
         AutoMLProgress<BinaryClassificationMetrics>.ReportCrossValidationRunDetails reportCrossValidationRunDetails = null) =>
         new AutoMLProgress<BinaryClassificationMetrics>(ml, contextName, reportRunDetails, reportCrossValidationRunDetails);
      /// <summary>
      /// Restituisce un progress per l'autotraining di clustering
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="contextName">Nome del contesto</param>
      /// <param name="reportRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<ClusteringMetrics> ClusteringProgress(
         this MLContext ml,
         string contextName = null,
         AutoMLProgress<ClusteringMetrics>.ReportRunDetails reportRunDetails = null,
         AutoMLProgress<ClusteringMetrics>.ReportCrossValidationRunDetails reportCrossValidationRunDetails = null) =>
         new AutoMLProgress<ClusteringMetrics>(ml, contextName, reportRunDetails, reportCrossValidationRunDetails);
      /// <summary>
      /// Restituisce un progress per l'autotraining di classificazione multi classe
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="contextName">Nome del contesto</param>
      /// <param name="reportRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<MulticlassClassificationMetrics> MulticlassClassificationProgress(
         this MLContext ml,
         string contextName = null,
         AutoMLProgress<MulticlassClassificationMetrics>.ReportRunDetails reportRunDetails = null,
         AutoMLProgress<MulticlassClassificationMetrics>.ReportCrossValidationRunDetails reportCrossValidationRunDetails = null) =>
         new AutoMLProgress<MulticlassClassificationMetrics>(ml, contextName, reportRunDetails, reportCrossValidationRunDetails);
      /// <summary>
      /// Restituisce un progress per l'autotraining di ranking
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="contextName">Nome del contesto</param>
      /// <param name="report">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<RankingMetrics> RankingProgress(
         this MLContext ml,
         string contextName = null,
         AutoMLProgress<RankingMetrics>.ReportRunDetails report = null,
         AutoMLProgress<RankingMetrics>.ReportCrossValidationRunDetails reportCrossValidationRunDetails = null) =>
         new AutoMLProgress<RankingMetrics>(ml, contextName, report, reportCrossValidationRunDetails);
      /// <summary>
      /// Restituisce un progress per l'autotraining di regressione
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="contextName">Nome del contesto</param>
      /// <param name="report">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<RegressionMetrics> RegressionProgress(
         this MLContext ml,
         string contextName = null,
         AutoMLProgress<RegressionMetrics>.ReportRunDetails report = null,
         AutoMLProgress<RegressionMetrics>.ReportCrossValidationRunDetails reportCrossValidationRunDetails = null) =>
         new AutoMLProgress<RegressionMetrics>(ml, contextName, report, reportCrossValidationRunDetails);
      /// <summary>
      /// Emette un log
      /// </summary>
      /// <param name="ml">Contesto ML</param>
      /// <param name="text">Testo di log</param>
      /// <param name="contextDescription">Descrizione del contesto</param>
      /// <param name="sensitivity">Sensitivita'</param>
      public static void WriteLog(this MLContext ml, string text, string contextDescription = null, MessageSensitivity sensitivity = MessageSensitivity.Unknown)
      {
         try {
            var channelProvider = (IChannelProvider)ml;
            using var channel = channelProvider.Start(contextDescription ?? channelProvider.ContextDescription);
            using var reader = new StringReader(text);
            for (var line = reader.ReadLine(); line != null; line = reader.ReadLine()) {
               if (!string.IsNullOrEmpty(line))
                  channel.Info(sensitivity, line);
            }
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
   }
}
