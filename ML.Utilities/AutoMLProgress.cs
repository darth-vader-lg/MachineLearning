using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Runtime;
using System;
using System.Diagnostics;

namespace ML.Utilities
{
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
}
