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
      /// Carica un modello
      /// </summary>
      /// <param name="path">Path del modello</param>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(string path, out DataViewSchema inputSchema)
      {
         LogAppendLine($"============== Loading the model  ===============");
         var result = Context.Model.Load(path, out inputSchema);
         LogAppendLine($"The model is saved to {path}");
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
      /// Funzione di aggiunta linea al log
      /// </summary>
      /// <param name="text">Testo da stampare</param>
      /// <param name="kind">Tipo di messaggio</param>
      public void LogAppendLine(string text, ChannelMessageKind kind = ChannelMessageKind.Info)
      {
         LogAppend((text ?? "") + Environment.NewLine, kind);
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
      /// Salva un modello
      /// </summary>
      /// <param name="model">Modello</param>
      /// <param name="schema">Schema dei dati</param>
      /// <param name="path">Path di destinazione</param>
      public void SaveModel(ITransformer model, DataViewSchema schema, string path)
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

   /// <summary>
   /// Estensioni
   /// </summary>
   public static class Extension
   {
      #region Methods
      /// <summary>
      /// Calcola l'intervallo di fiducia 95%
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>L'intervallo di fiducia 95%</returns>
      public static double CalculateConfidenceInterval95(IEnumerable<double> values)
      {
         var confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
         return confidenceInterval95;
      }
      /// <summary>
      /// Calcola la deviazione standard di un set di valori
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>La deviazione standard</returns>
      public static double CalculateStandardDeviation(IEnumerable<double> values)
      {
         var average = values.Average();
         var sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
         var standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
         return standardDeviation;
      }
      /// <summary>
      /// Valutazione incrociata di un modello a classificazione multipla
      /// </summary>
      /// <param name="catalog">Catalogo</param>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">Dati</param>
      /// <param name="estimator">Pipeline</param>
      /// <param name="numberOfFolds">Numero di valutazioni</param>
      /// <param name="labelColumnName">Nome della colonna di previsione</param>
      /// <param name="samplingKeyColumnName">Nome colonna di campionamento</param>
      /// <param name="seed">Seme generatore numeri casuali</param>
      /// <returns>Il modello migliore</returns>
      public static ITransformer CrossValidate(
         this MulticlassClassificationCatalog catalog,
         ML ml,
         IDataView data,
         IEstimator<ITransformer> estimator,
         int numberOfFolds = 5,
         string labelColumnName = "Label",
         string samplingKeyColumnName = null,
         int? seed = null)
      {
         // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
         // in order to evaluate and get the model's accuracy metrics
         ml.LogAppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
         var crossValidationResults = catalog.CrossValidate(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
         Print(ml, crossValidationResults);
         var result = (from fold in crossValidationResults
                       orderby fold.Metrics.LogLoss
                       select fold.Model).First();
         return result;
      }
      /// <summary>
      /// Valutazione incrociata di un modello a regressione
      /// </summary>
      /// <param name="catalog">Catalogo</param>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">Dati</param>
      /// <param name="estimator">Pipeline</param>
      /// <param name="numberOfFolds">Numero di valutazioni</param>
      /// <param name="labelColumnName">Nome della colonna di previsione</param>
      /// <param name="samplingKeyColumnName">Nome colonna di campionamento</param>
      /// <param name="seed">Seme generatore numeri casuali</param>
      /// <returns>Il modello migliore</returns>
      public static ITransformer CrossValidate(
         this RegressionCatalog catalog,
         ML ml,
         IDataView data,
         IEstimator<ITransformer> estimator,
         int numberOfFolds = 5,
         string labelColumnName = "Label",
         string samplingKeyColumnName = null,
         int? seed = null)
      {
         // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
         // in order to evaluate and get the model's accuracy metrics
         ml.LogAppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
         var crossValidationResults = catalog.CrossValidate(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
         Print(ml, crossValidationResults);
         var result = (from fold in crossValidationResults
                       orderby fold.Metrics.LossFunction
                       select fold.Model).First();
         return result;
      }
      /// <summary>
      /// Valuta un modello a classificazione multipla
      /// </summary>
      /// <param name="catalog">Catalogo</param>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">Dati</param>
      /// <param name="labelColumnName">Nome della colonna della label</param>
      /// <param name="scoreColumnName">Nome colonna del punteggio</param>
      /// <param name="predictedLabelColumnName">Nome colonna di previsione</param>
      /// <param name="topKPredictionCount"></param>
      public static void Evaluate(
         this MulticlassClassificationCatalog catalog,
         ML ml,
         IDataView data,
         string labelColumnName = "Label",
         string scoreColumnName = "Score",
         string predictedLabelColumnName = "Score",
         int topKPredictionCount = 0)
      {
         // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
         // in order to evaluate and get the model's accuracy metrics
         ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
         var metrics = catalog.Evaluate(data, labelColumnName, scoreColumnName, predictedLabelColumnName, topKPredictionCount);
         Print(ml, metrics);
      }
      /// <summary>
      /// Valuta un modello di regressione
      /// </summary>
      /// <param name="catalog">Catalogo</param>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">Dati</param>
      /// <param name="labelColumnName">Nome della colonna della label</param>
      /// <param name="scoreColumnName">Nome colonna del punteggio</param>
      public static void Evaluate(
         this RegressionCatalog catalog,
         ML ml,
         IDataView data,
         string labelColumnName = "Label",
         string scoreColumnName = "Score")
      {
         // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
         // in order to evaluate and get the model's accuracy metrics
         ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
         var metrics = catalog.Evaluate(data, labelColumnName, scoreColumnName);
         Print(ml, metrics);
      }
      /// <summary>
      /// Stampa la metrica media di un modello a classificazione multipla
      /// </summary>
      /// <param name="crossValResults">Risultati di una validazione incrociata</param>
      public static void Print(ML ml, IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
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
            sb.AppendLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:0.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:0.###})");
            sb.AppendLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:0.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:0.###})");
            sb.AppendLine($"*       Average LogLoss:          {logLossAverage:0.###}  - Standard deviation: ({logLossStdDeviation:0.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:0.###})");
            sb.AppendLine($"*       Average LogLossReduction: {logLossReductionAverage:0.###}  - Standard deviation: ({logLossReductionStdDeviation:0.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:0.###})");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Stampa la metrica media di un modello di regressione
      /// </summary>
      /// <param name="crossValResults">Risultati di una validazione incrociata</param>
      private static void Print(ML ml, IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValResults)
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
            ml.LogAppend(sb.ToString());
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Stampa la metrica media di un modello a classificazione multipla
      /// </summary>
      /// <param name="crossValResults">Risultati di una validazione incrociata</param>
      public static void Print(ML ml, MulticlassClassificationMetrics metrics)
      {
         try {
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Multi-class Classification model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            sb.AppendLine($"*       MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            sb.AppendLine($"*       LogLoss:          {metrics.LogLoss:0.###}");
            sb.AppendLine($"*       LogLossReduction: {metrics.LogLossReduction:0.###}");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Stampa la metrica media di un modello di regressione
      /// </summary>
      /// <param name="crossValResults">Risultati di una validazione incrociata</param>
      public static void Print(ML ml, RegressionMetrics metrics)
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
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }
}
