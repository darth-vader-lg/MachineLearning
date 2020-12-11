using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MachineLearningStudio
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
      /// Restituisce un progress per l'autotraining di classificazione binaria
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="report">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<BinaryClassificationMetrics> BinaryClassificationProgress(this MLContext ml, AutoMLProgress<BinaryClassificationMetrics>.Report report = null) =>
         new AutoMLProgress<BinaryClassificationMetrics>(ml, report);
      /// <summary>
      /// Restituisce un progress per l'autotraining di classificazione multi classe
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="report">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<MulticlassClassificationMetrics> MulticlassClassificationProgress(this MLContext ml, AutoMLProgress<MulticlassClassificationMetrics>.Report report = null) =>
         new AutoMLProgress<MulticlassClassificationMetrics>(ml, report);
      /// <summary>
      /// Restituisce un progress per l'autotraining di ranking
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="report">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<RankingMetrics> RankingProgress(this MLContext ml, AutoMLProgress<RankingMetrics>.Report report = null) =>
         new AutoMLProgress<RankingMetrics>(ml, report);
      /// <summary>
      /// Restituisce un progress per l'autotraining di regressione
      /// </summary>
      /// <param name="ml">Contesto ml</param>
      /// <param name="report">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <returns>La classe di progress</returns>
      public static AutoMLProgress<RegressionMetrics> RegressionProgress(this MLContext ml, AutoMLProgress<RegressionMetrics>.Report report = null) =>
         new AutoMLProgress<RegressionMetrics>(ml, report);
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
            var channel = channelProvider.Start(contextDescription ?? channelProvider.ContextDescription);
            using var reader = new StringReader(text);
            for (var line = reader.ReadLine(); line != null; line = reader.ReadLine())
               channel.Info(sensitivity, line);
            channel.Dispose();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
   }

   /// <summary>
   /// Gestore del progress di auto machine learning
   /// </summary>
   /// <typeparam name="TMetrics">Tipo di metrica</typeparam>
   public class AutoMLProgress<TMetrics> : IProgress<RunDetail<TMetrics>>
   {
      #region Fields
      /// <summary>
      /// Contesto ML
      /// </summary>
      private readonly MLContext ml;
      /// <summary>
      /// Delegato al report
      /// </summary>
      private readonly Report report;
      #endregion
      #region Events and delegates
      /// <summary>
      /// Delegato di report dei progressi
      /// </summary>
      /// <param name="sender">Oggetto di progress generante</param>
      /// <param name="e">Argomenti del progress</param>
      public delegate void Report(AutoMLProgress<TMetrics> sender, RunDetail<TMetrics> e);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto ML</param>
      /// <param name="report">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      public AutoMLProgress(MLContext ml, Report report = null)
      {
         this.ml = ml;
         this.report = report;
      }
      /// <summary>
      /// Funzione di report del progresso
      /// </summary>
      /// <param name="runDetail">Dettagli del report</param>
      void IProgress<RunDetail<TMetrics>>.Report(RunDetail<TMetrics> runDetail)
      {
         if (report != null)
            report(this, runDetail);
         else
            WriteLog(runDetail);
      }
      /// <summary>
      /// Riporta un messaggio di log di AutoML
      /// </summary>
      /// <param name="value">Dettagli del report</param>
      public void WriteLog(RunDetail<TMetrics> runDetail)
      {
         try {
            if (runDetail.Exception == null)
               ml.WriteLog($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs", "AutoML", MessageSensitivity.Unknown);
            else
               ml.WriteLog($"Exception: {runDetail.Exception.Message}", "AutoML", MessageSensitivity.Unknown);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }

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

   /// <summary>
   /// Estensioni della categoria di clusterizzazione
   /// </summary>
   public static class ClusteringExtensions
   {
      /// <summary>
      /// Estrae il migliore da un elenco di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il risultato migliore</returns>
      public static TrainCatalogBase.CrossValidationResult<ClusteringMetrics> Best(this IEnumerable<TrainCatalogBase.CrossValidationResult<ClusteringMetrics>> results)
      {
         try {
            var result = (from item in results
                          orderby item.Metrics.AverageDistance
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
            sb.AppendLine($"*       Average NormalizedMutualInformation: {normalizedMutualInformation.Average():0.###}  ");
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

   /// <summary>
   /// Estensioni della categoria di classificazione multiclasse
   /// </summary>
   public static class MulticlassClassificationExtensions
   {
      /// <summary>
      /// Estrae il migliore da un elenco di risultati di validazione incrociata
      /// </summary>
      /// <param name="results">Elenco di risultati</param>
      /// <returns>Il risultato migliore</returns>
      public static TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics> Best(this IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> results)
      {
         try {
            var result = (from item in (from item in results
                                        group item by item.Metrics.MicroAccuracy into grps
                                        orderby grps.Key descending
                                        select grps).First()
                          orderby item.Metrics.LogLoss
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
