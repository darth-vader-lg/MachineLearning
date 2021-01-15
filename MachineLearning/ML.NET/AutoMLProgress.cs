using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Diagnostics;

namespace Microsoft.ML.AutoML
{
   /// <summary>
   /// Gestore del progress di auto machine learning
   /// </summary>
   /// <typeparam name="TMetrics">Tipo di metrica</typeparam>
   public class AutoMLProgress<TMetrics> : Progress<CrossValidationRunDetail<TMetrics>>, IProgress<RunDetail<TMetrics>>
   {
      #region Fields
      /// <summary>
      /// Nome del contesto
      /// </summary>
      private readonly string contextName;
      /// <summary>
      /// Contesto ML
      /// </summary>
      private readonly MLContext ml;
      /// <summary>
      /// Delegato al report
      /// </summary>
      private readonly ReportRunDetails reportRunDetails;
      /// <summary>
      /// Delegato al report
      /// </summary>
      private readonly ReportCrossValidationRunDetails reportCrossValidationRunDetails;
      #endregion
      #region Events and delegates
      /// <summary>
      /// Delegato di report dei progressi
      /// </summary>
      /// <param name="sender">Oggetto di progress generante</param>
      /// <param name="e">Argomenti del progress</param>
      public delegate void ReportRunDetails(AutoMLProgress<TMetrics> sender, RunDetail<TMetrics> e);
      /// <summary>
      /// Delegato di report dei progressi
      /// </summary>
      /// <param name="sender">Oggetto di progress generante</param>
      /// <param name="e">Argomenti del progress</param>
      public delegate void ReportCrossValidationRunDetails(AutoMLProgress<TMetrics> sender, CrossValidationRunDetail<TMetrics> e);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto ML</param>
      /// <param name="contextName">Nome del contesto</param>
      /// <param name="reportRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      public AutoMLProgress(MLContext ml, string contextName, ReportRunDetails reportRunDetails = null, ReportCrossValidationRunDetails reportCrossValidationRunDetails = null)
      {
         this.ml = ml;
         this.contextName = contextName ?? "AutoML";
         this.reportRunDetails = reportRunDetails;
         this.reportCrossValidationRunDetails = reportCrossValidationRunDetails;
      }
      /// <summary>
      /// Funzione di report del progresso
      /// </summary>
      /// <param name="value">Dettagli del report</param>
      protected override void OnReport(CrossValidationRunDetail<TMetrics> value)
      {
         if (reportCrossValidationRunDetails != null)
            reportCrossValidationRunDetails(this, value);
         else
            WriteLog(value);
      }
      /// <summary>
      /// Funzione di report del progresso
      /// </summary>
      /// <param name="runDetail">Dettagli del report</param>
      void IProgress<RunDetail<TMetrics>>.Report(RunDetail<TMetrics> runDetail)
      {
         if (reportRunDetails != null)
            reportRunDetails(this, runDetail);
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
            if (runDetail.Exception == null) {
               ml.WriteLog($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs", contextName, MessageSensitivity.Unknown);
               WriteValidationMetrics(runDetail.ValidationMetrics);
            }
            else
               ml.WriteLog($"Exception: {runDetail.Exception.Message}", contextName, MessageSensitivity.Unknown);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Riporta un messaggio di log di AutoML
      /// </summary>
      /// <param name="value">Dettagli del report</param>
      public void WriteLog(CrossValidationRunDetail<TMetrics> runDetail)
      {
         try {
            ml.WriteLog($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs", contextName, MessageSensitivity.Unknown);
            foreach (var result in runDetail.Results) {
               if (result.Exception == null)
                  WriteValidationMetrics(result.ValidationMetrics);
               else
                  ml.WriteLog($"Exception: {result.Exception.Message}", contextName, MessageSensitivity.Unknown);
            }
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Riporta un messaggio di metrica nel log
      /// </summary>
      /// <param name="metrics">Metriche</param>
      protected void WriteValidationMetrics(TMetrics metrics)
      {
         try {
            if (metrics is AnomalyDetectionMetrics anomaly)
               ml.WriteLog(anomaly.ToText(), contextName, MessageSensitivity.Unknown);
            else if (metrics is BinaryClassificationMetrics binary)
               ml.WriteLog(binary.ToText(), contextName, MessageSensitivity.Unknown);
            else if (metrics is ClusteringMetrics clustering)
               ml.WriteLog(clustering.ToText(), contextName, MessageSensitivity.Unknown);
            else if (metrics is MulticlassClassificationMetrics multiclass)
               ml.WriteLog(multiclass.ToText(), contextName, MessageSensitivity.Unknown);
            else if (metrics is RankingMetrics ranking)
               ml.WriteLog(ranking.ToText(), contextName, MessageSensitivity.Unknown);
            else if (metrics is RegressionMetrics regression)
               ml.WriteLog(regression.ToText(), contextName, MessageSensitivity.Unknown);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }
}
