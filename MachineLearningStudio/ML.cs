using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MachineLearningStudio
{
   /// <summary>
   /// Classe di utilità generiche
   /// </summary>
   public partial class ML
   {
      #region Fields
      /// <summary>
      /// Buffer di log
      /// </summary>
      private readonly StringBuilder logBuilder = new StringBuilder();
      #endregion
      #region Properties
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      public MLContext Context { get; }
      /// <summary>
      /// Log
      /// </summary>
      public string Log { get { lock (logBuilder) return logBuilder.ToString(); } }
      #endregion
      #region Events and delegates
      /// <summary>
      /// Evento messaggio di log
      /// </summary>
      public event MLLogEventHandler LogMessage;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme del generatore di numeri casuali</param>
      public ML(int? seed = null)
      {
         Context = new MLContext(seed);
         Context.Log += Context_Log;
      }
      /// <summary>
      /// Calcola l'intervallo di fiducia 95%
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>L'intervallo di fiducia 95%</returns>
      protected double CalculateConfidenceInterval95(IEnumerable<double> values)
      {
         var confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
         return confidenceInterval95;
      }
      /// <summary>
      /// Calcola la deviazione standard di un set di valori
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>La deviazione standard</returns>
      protected double CalculateStandardDeviation(IEnumerable<double> values)
      {
         var average = values.Average();
         var sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
         var standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
         return standardDeviation;
      }
      /// <summary>
      /// Evento di log del contesto ML
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void Context_Log(object sender, LoggingEventArgs e)
      {
         try {
            LogAppendLine(e.Message, e.Kind);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Valuta un modello a classificazione multipla
      /// </summary>
      /// <param name="trainingDataView">Dati</param>
      /// <param name="trainingPipeline">Pipeline</param>
      /// <param name="labelColumnName">Nome della colonna di previsione</param>
      /// <param name="numberOfFolds">Numero di valutazioni</param>
      /// <returns>Il modello migliore</returns>
      public ITransformer EvaluateMulticlassClassification(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline, string labelColumnName, int numberOfFolds = 5)
      {
         // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
         // in order to evaluate and get the model's accuracy metrics
         LogAppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
         var crossValidationResults = Context.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds, labelColumnName);
         PrintMulticlassClassificationFoldsAverageMetrics(crossValidationResults);
         var result = (from fold in crossValidationResults
                       orderby fold.Metrics.LogLoss
                       select fold.Model).First();
         return result;
      }
      /// <summary>
      /// Valuta un modello a classificazione multipla
      /// </summary>
      /// <param name="trainingDataView">Dati</param>
      /// <param name="trainingPipeline">Pipeline</param>
      /// <param name="labelColumnName">Nome della colonna di previsione</param>
      /// <param name="numberOfFolds">Numero di valutazioni</param>
      /// <returns>Il modello migliore</returns>
      public ITransformer EvaluateRegression(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline, string labelColumnName, int numberOfFolds = 5)
      {
         // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
         // in order to evaluate and get the model's accuracy metrics
         LogAppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
         var crossValidationResults = Context.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds, labelColumnName);
         PrintRegressionFoldsAverageMetrics(crossValidationResults);
         var result = (from fold in crossValidationResults
                       orderby fold.Metrics.LossFunction
                       select fold.Model).First();
         return result;
      }
      /// <summary>
      /// Funzione di aggiunta al log
      /// </summary>
      /// <param name="text">Testo da stampare</param>
      /// <param name="kind">Tipo di messaggio</param>
      public void LogAppend(string text, ChannelMessageKind kind = ChannelMessageKind.Info)
      {
         try {
            if (string.IsNullOrEmpty(text))
               return;
            lock (logBuilder)
               logBuilder.Append(text);
            OnLogMessage(new MLLogMessageEventArgs(text, kind));
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di logging dei messaggi
      /// </summary>
      /// <param name="mLLogMessageEventArgs">Argomenti del log</param>
      protected virtual void OnLogMessage(MLLogMessageEventArgs e)
      {
         try {
            LogMessage?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di aggiunta linea al log
      /// </summary>
      /// <param name="text">Testo da stampare</param>
      /// <param name="kind">Tipo di messaggio</param>
      public void LogAppendLine(string text, ChannelMessageKind kind = ChannelMessageKind.Info)
      {
         LogAppend((text ?? "") + Environment.NewLine, kind);
      }
      /// <summary>
      /// Stampa la metrica media di un modello a classificazione multipla
      /// </summary>
      /// <param name="crossValResults">Risultati di una validazione incrociata</param>
      protected void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
      {
         try {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);
            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);
            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);
            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);
            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Multi-class Classification model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            sb.AppendLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            sb.AppendLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            sb.AppendLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            sb.AppendLine($"*************************************************************************************************************");
            LogAppend(sb.ToString());
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Stampa la metrica media di un modello a classificazione multipla
      /// </summary>
      /// <param name="crossValResults">Risultati di una validazione incrociata</param>
      protected void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValResults)
      {
         try {
            var L1 = crossValResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValResults.Select(r => r.Metrics.RSquared);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Regression model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            sb.AppendLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            sb.AppendLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            sb.AppendLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            sb.AppendLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            sb.AppendLine($"*************************************************************************************************************");
            LogAppend(sb.ToString());
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Salva un modello
      /// </summary>
      /// <param name="schema">Schema dei dati</param>
      /// <param name="model">Modello</param>
      /// <param name="path">Path di destinazione</param>
      public void SaveModel(DataViewSchema schema, ITransformer model, string path)
      {
         // Save/persist the trained model to a .ZIP file
         LogAppendLine($"=============== Saving the model  ===============");
         Context.Model.Save(model, schema, path);
         LogAppendLine($"The model is saved to {path}");
      }
      /// <summary>
      /// Funzione di training del modello
      /// </summary>
      /// <param name="trainingDataView">Dati di training</param>
      /// <param name="trainingPipeline">Pipeline di training</param>
      /// <returns></returns>
      public ITransformer TrainModel(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
      {
         LogAppendLine("=============== Training  model ===============");
         var model = trainingPipeline.Fit(trainingDataView);
         LogAppendLine("=============== End of training process ===============");
         return model;
      }
      #endregion
   }
}
