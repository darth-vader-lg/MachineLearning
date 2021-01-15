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
      /// <param name="reportRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      public AutoMLProgress(MLContext ml, ReportRunDetails reportRunDetails = null, ReportCrossValidationRunDetails reportCrossValidationRunDetails = null)
      {
         this.ml = ml;
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
            if (runDetail.Exception == null)
               ml.WriteLog($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs", "AutoML", MessageSensitivity.Unknown);
            else
               ml.WriteLog($"Exception: {runDetail.Exception.Message}", "AutoML", MessageSensitivity.Unknown);
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
            ml.WriteLog($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs", "AutoML", MessageSensitivity.Unknown);
            foreach (var result in runDetail.Results) {
               if (result.Exception == null) {
                  if (result.ValidationMetrics is AnomalyDetectionMetrics anomaly)
                     ml.WriteLog(anomaly.ToText());
                  else if (result.ValidationMetrics is BinaryClassificationMetrics binary)
                     ml.WriteLog(binary.ToText());
                  else if (result.ValidationMetrics is ClusteringMetrics clustering)
                     ml.WriteLog(clustering.ToText());
                  else if (result.ValidationMetrics is MulticlassClassificationMetrics multiclass)
                     ml.WriteLog(multiclass.ToText());
                  else if (result.ValidationMetrics is RankingMetrics ranking)
                     ml.WriteLog(ranking.ToText());
                  else if (result.ValidationMetrics is RegressionMetrics regression)
                     ml.WriteLog(regression.ToText());
               }
               else
                  ml.WriteLog($"Exception: {result.Exception.Message}", "AutoML", MessageSensitivity.Unknown);
            }
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }
}
