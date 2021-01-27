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
   public class AutoMLProgress<TMetrics> : Progress<CrossValidationRunDetail<TMetrics>>, IDisposable, IProgress<RunDetail<TMetrics>>
   {
      #region Fields
      /// <summary>
      /// Canale di messaggistica
      /// </summary>
      private IChannel channel;
      /// <summary>
      /// Contesto ML
      /// </summary>
      private readonly IChannelProvider context;
      /// <summary>
      /// Indicatore di oggetto disposed
      /// </summary>
      private bool disposed;
      /// <summary>
      /// Delegato al report
      /// </summary>
      private readonly ReportRunDetails reportRunDetails;
      /// <summary>
      /// Delegato al report
      /// </summary>
      private readonly ReportCrossValidationRunDetails reportCrossValidationRunDetails;
      #endregion
      #region Properties
      /// <summary>
      /// Canale di messaggistica
      /// </summary>
      protected IChannel Channel => channel ??= context.Start(context.ContextDescription);
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
      /// <param name="context">Contesto</param>
      /// <param name="reportRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      /// <param name="reportCrossValidationRunDetails">Delegato al report. Se null il report viene loggato nel contesto ML</param>
      public AutoMLProgress(IChannelProvider context, ReportRunDetails reportRunDetails = null, ReportCrossValidationRunDetails reportCrossValidationRunDetails = null)
      {
         Contracts.CheckValue(this.context = context, nameof(context));
         this.reportRunDetails = reportRunDetails;
         this.reportCrossValidationRunDetails = reportCrossValidationRunDetails;
      }
      /// <summary>
      /// Dispose da programma
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di Dispose da programma</param>
      protected virtual void Dispose(bool disposing)
      {
         if (!disposed) {
            if (disposing) {
               channel?.Dispose();
               channel = null;
            }
            disposed = true;
         }
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
               Channel.WriteLog($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs");
               WriteValidationMetrics(runDetail.ValidationMetrics);
            }
            else
               Channel.WriteLog($"Exception: {runDetail.Exception.Message}");
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
            Channel.WriteLog($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs");
            foreach (var result in runDetail.Results) {
               if (result.Exception == null)
                  WriteValidationMetrics(result.ValidationMetrics);
               else
                  Channel.WriteLog($"Exception: {result.Exception.Message}");
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
               Channel.WriteLog(anomaly.ToText());
            else if (metrics is BinaryClassificationMetrics binary)
               Channel.WriteLog(binary.ToText());
            else if (metrics is ClusteringMetrics clustering)
               Channel.WriteLog(clustering.ToText());
            else if (metrics is MulticlassClassificationMetrics multiclass)
               Channel.WriteLog(multiclass.ToText());
            else if (metrics is RankingMetrics ranking)
               Channel.WriteLog(ranking.ToText());
            else if (metrics is RegressionMetrics regression)
               Channel.WriteLog(regression.ToText());
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }
}
