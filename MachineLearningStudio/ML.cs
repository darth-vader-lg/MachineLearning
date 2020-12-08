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
      /// Trainers e tasks specifici dei problemi di classificazione binaria.
      /// </summary>
      public BinaryClassificationCatalog BinaryClassification { get; }
      /// <summary>
      /// Trainers e tasks specifici dei problemi di clusterizzazione.
      /// </summary>
      public ClusteringCatalog Clustering { get; }
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      public MLContext Context { get; }
      /// <summary>
      /// Log
      /// </summary>
      public string Log { get { lock (logBuilder) return logBuilder.ToString(); } }
      /// <summary>
      /// Trainers e tasks specifici dei problemi di classificazione multi classe.
      /// </summary>
      public MulticlassClassificationCatalog MulticlassClassification { get; }
      /// <summary>
      /// Trainers e tasks specifici dei problemi di ranking.
      /// </summary>
      public RankingCatalog Ranking { get; }
      /// <summary>
      /// Trainers e tasks specifici dei problemi di regressione.
      /// </summary>
      public RegressionCatalog Regression { get; }
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
         BinaryClassification = new BinaryClassificationCatalog(this);
         Clustering = new ClusteringCatalog(this);
         MulticlassClassification = new MulticlassClassificationCatalog(this);
         Ranking = new RankingCatalog(this);
         Regression = new RegressionCatalog(this);
      }
      /// <summary>
      /// Calcola l'intervallo di fiducia 95%
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>L'intervallo di fiducia 95%</returns>
      protected static double CalculateConfidenceInterval95(IEnumerable<double> values)
      {
         var confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
         return confidenceInterval95;
      }
      /// <summary>
      /// Calcola la deviazione standard di un set di valori
      /// </summary>
      /// <param name="values">Set di valori</param>
      /// <returns>La deviazione standard</returns>
      protected static double CalculateStandardDeviation(IEnumerable<double> values)
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
      /// Carica un modello
      /// </summary>
      /// <param name="path">Path del modello</param>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(string path, out DataViewSchema inputSchema)
      {
         LogAppendLine($"============== Loading the model  ===============");
         var result = Context.Model.Load(path, out inputSchema);
         LogAppendLine($"The model is loaded from {path}");
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
   /// Catalogo classificazione binaria
   /// </summary>
   public partial class ML
   {
      public sealed class BinaryClassificationCatalog
      {
         #region Fields
         /// <summary>
         /// Owner
         /// </summary>
         private readonly ML ml;
         #endregion
         #region Properties
         /// <summary>
         /// The list of calibrators
         /// </summary>
         public Microsoft.ML.BinaryClassificationCatalog.CalibratorsCatalog Calibrators => ml.Context.BinaryClassification.Calibrators;
         /// <summary>
         /// The list of trainers
         /// </summary>
         public Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers Trainers => ml.Context.BinaryClassification.Trainers;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="ml">Owner</param>
         public BinaryClassificationCatalog(ML ml) => this.ml = ml;
         /// <summary>
         /// Method to modify the threshold to existing model and return modified model.
         /// </summary>
         /// <typeparam name="TModel">The type of the model parameters.</typeparam>
         /// <param name="model">Existing model to modify threshold.</param>
         /// <param name="threshold">New threshold.</param>
         /// <returns>New model with modified threshold.</returns>
         public BinaryPredictionTransformer<TModel> ChangeModelThreshold<TModel>(BinaryPredictionTransformer<TModel> model, float threshold) where TModel : class =>
            ml.Context.BinaryClassification.ChangeModelThreshold(model, threshold);
         /// <summary>
         /// Run cross-validation over numberOfFolds folds of data, by fitting estimator,
         /// and respecting samplingKeyColumnName if provided. Then evaluate each sub-model
         /// against labelColumnName and return metrics.
         /// </summary>
         /// <param name="data">The data to run cross-validation on.</param>
         /// <param name="estimator">The estimator to fit.</param>
         /// <param name="numberOfFolds">Number of cross-validation folds.</param>
         /// <param name="labelColumnName">Optional label column for evaluation (clustering tasks may not always have a label.</param>
         /// <param name="samplingKeyColumnName">
         /// Name of a column to use for grouping rows. If two examples share the same value
         /// of the samplingKeyColumnName, they are guaranteed to appear in the same subset
         /// (train or test). This can be used to ensure no label leakage from the train to
         /// the test set. If null no row grouping will be performed.
         /// </param>
         /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
         /// <returns>The best model</returns>
         public ITransformer CrossValidate(
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
            var crossValidationResults = ml.Context.BinaryClassification.CrossValidate(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
            var accuracy = crossValidationResults.Select(r => r.Metrics.Accuracy);
            var areaUnderPrecisionRecallCurve = crossValidationResults.Select(r => r.Metrics.AreaUnderPrecisionRecallCurve);
            var areaUnderRocCurve = crossValidationResults.Select(r => r.Metrics.AreaUnderRocCurve);
            var entropy = crossValidationResults.Select(r => r.Metrics.Entropy);
            var F1Score = crossValidationResults.Select(r => r.Metrics.F1Score);
            var logLoss = crossValidationResults.Select(r => r.Metrics.LogLoss);
            var logLossReduction = crossValidationResults.Select(r => r.Metrics.LogLossReduction);
            var negativePrecision = crossValidationResults.Select(r => r.Metrics.NegativePrecision);
            var negativeRecall = crossValidationResults.Select(r => r.Metrics.NegativeRecall);
            var positivePrecision = crossValidationResults.Select(r => r.Metrics.PositivePrecision);
            var positiveRecall = crossValidationResults.Select(r => r.Metrics.PositiveRecall);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Binary classification model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
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
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            var result = (from fold in crossValidationResults
                          orderby fold.Metrics.Accuracy descending
                          select fold.Model).First();
            return result;
         }
         /// <summary>
         /// Run cross-validation over numberOfFolds folds of data, by fitting estimator,
         /// and respecting samplingKeyColumnName if provided. Then evaluate each sub-model
         /// against labelColumnName and return metrics.
         /// </summary>
         /// <param name="data">The data to run cross-validation on.</param>
         /// <param name="estimator">The estimator to fit.</param>
         /// <param name="numberOfFolds">Number of cross-validation folds.</param>
         /// <param name="labelColumnName">Optional label column for evaluation (clustering tasks may not always have a label.</param>
         /// <param name="samplingKeyColumnName">
         /// Name of a column to use for grouping rows. If two examples share the same value
         /// of the samplingKeyColumnName, they are guaranteed to appear in the same subset
         /// (train or test). This can be used to ensure no label leakage from the train to
         /// the test set. If null no row grouping will be performed.
         /// </param>
         /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
         /// <returns>The best model</returns>
         public ITransformer CrossValidateNonCalibrated(
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
            var crossValidationResults = ml.Context.BinaryClassification.CrossValidateNonCalibrated(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
            var accuracy = crossValidationResults.Select(r => r.Metrics.Accuracy);
            var areaUnderPrecisionRecallCurve = crossValidationResults.Select(r => r.Metrics.AreaUnderPrecisionRecallCurve);
            var areaUnderRocCurve = crossValidationResults.Select(r => r.Metrics.AreaUnderRocCurve);
            var F1Score = crossValidationResults.Select(r => r.Metrics.F1Score);
            var negativePrecision = crossValidationResults.Select(r => r.Metrics.NegativePrecision);
            var negativeRecall = crossValidationResults.Select(r => r.Metrics.NegativeRecall);
            var positivePrecision = crossValidationResults.Select(r => r.Metrics.PositivePrecision);
            var positiveRecall = crossValidationResults.Select(r => r.Metrics.PositiveRecall);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Binary classification model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       Average Accuracy:                      {accuracy.Average():0.###} ");
            sb.AppendLine($"*       Average AreaUnderPrecisionRecallCurve: {areaUnderPrecisionRecallCurve.Average():0.###}  ");
            sb.AppendLine($"*       Average AreaUnderRocCurve:             {areaUnderRocCurve.Average():0.###}  ");
            sb.AppendLine($"*       Average F1Score:                       {F1Score.Average():0.###}  ");
            sb.AppendLine($"*       Average NegativePrecision:             {negativePrecision.Average():0.###}  ");
            sb.AppendLine($"*       Average NegativeRecall:                {negativeRecall.Average():0.###}  ");
            sb.AppendLine($"*       Average PositivePrecision:             {positivePrecision.Average():0.###}  ");
            sb.AppendLine($"*       Average PositiveRecall:                {positiveRecall.Average():0.###}  ");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            var result = (from fold in crossValidationResults
                          orderby fold.Metrics.Accuracy descending
                          select fold.Model).First();
            return result;
         }
         /// <summary>
         /// Evaluates scored multiclass classification data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <param name="probabilityColumnName">The name of the probability column in data, the calibrated version of scoreColumnName.</param>
         /// <param name="predictedLabelColumnName">The name of the predicted label column in data.</param>
         /// <returns>The evaluation results for these calibrated outputs.</returns>
         public CalibratedBinaryClassificationMetrics Evaluate(
            IDataView data,
            string labelColumnName = "Label",
            string scoreColumnName = "Score",
            string probabilityColumnName = "Probability",
            string predictedLabelColumnName = "PredictedLabel")
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.BinaryClassification.Evaluate(data, labelColumnName, scoreColumnName, probabilityColumnName, predictedLabelColumnName);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Binary classification model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
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
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            return metrics;
         }
         /// <summary>
         /// Evaluates scored multiclass classification data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <param name="predictedLabelColumnName">The name of the predicted label column in data.</param>
         /// <returns>The evaluation results for these calibrated outputs.</returns>
         public BinaryClassificationMetrics EvaluateNonCalibrated(
            IDataView data,
            string labelColumnName = "Label",
            string scoreColumnName = "Score",
            string predictedLabelColumnName = "PredictedLabel")
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.BinaryClassification.EvaluateNonCalibrated(data, labelColumnName, scoreColumnName, predictedLabelColumnName);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Binary classification model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       Accuracy:                     {metrics.Accuracy:0.###} ");
            sb.AppendLine($"*       AreaUnderPrecisionRecallCurve:{metrics.AreaUnderPrecisionRecallCurve:0.###}  ");
            sb.AppendLine($"*       AreaUnderRocCurve:            {metrics.AreaUnderRocCurve:0.###}  ");
            sb.AppendLine($"*       F1Score:                      {metrics.F1Score:0.###}  ");
            sb.AppendLine($"*       NegativePrecision:            {metrics.NegativePrecision:0.###}  ");
            sb.AppendLine($"*       NegativeRecall:               {metrics.NegativeRecall:0.###}  ");
            sb.AppendLine($"*       PositivePrecision:            {metrics.PositivePrecision:0.###}  ");
            sb.AppendLine($"*       PositiveRecall:               {metrics.PositiveRecall:0.###}  ");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            return metrics;
         }
         #endregion
      }
   }

   /// <summary>
   /// Catalogo clasterizzazione
   /// </summary>
   public partial class ML
   {
      public sealed class ClusteringCatalog
      {
         #region Fields
         /// <summary>
         /// Owner
         /// </summary>
         private readonly ML ml;
         #endregion
         #region Properties
         /// <summary>
         /// The list of trainers
         /// </summary>
         public Microsoft.ML.ClusteringCatalog.ClusteringTrainers Trainers => ml.Context.Clustering.Trainers;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="ml"></param>
         public ClusteringCatalog(ML ml) => this.ml = ml;
         /// <summary>
         /// Run cross-validation over numberOfFolds folds of data, by fitting estimator,
         /// and respecting samplingKeyColumnName if provided. Then evaluate each sub-model
         /// against labelColumnName and return metrics.
         /// </summary>
         /// <param name="data">The data to run cross-validation on.</param>
         /// <param name="estimator">The estimator to fit.</param>
         /// <param name="numberOfFolds">Number of cross-validation folds.</param>
         /// <param name="labelColumnName">Optional label column for evaluation (clustering tasks may not always have a label.</param>
         /// <param name="featuresColumnName">Optional features column for evaluation (needed for calculating Dbi metric)</param>
         /// <param name="samplingKeyColumnName">
         /// Name of a column to use for grouping rows. If two examples share the same value
         /// of the samplingKeyColumnName, they are guaranteed to appear in the same subset
         /// (train or test). This can be used to ensure no label leakage from the train to
         /// the test set. If null no row grouping will be performed.
         /// </param>
         /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
         /// <returns>The best model</returns>
         public ITransformer CrossValidate(
            IDataView data,
            IEstimator<ITransformer> estimator,
            int numberOfFolds = 5,
            string labelColumnName = null,
            string featuresColumnName = null,
            string samplingKeyColumnName = null,
            int? seed = null)
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = ml.Context.Clustering.CrossValidate(data, estimator, numberOfFolds, labelColumnName, featuresColumnName, samplingKeyColumnName, seed);
            var averageDistance = crossValidationResults.Select(r => r.Metrics.AverageDistance);
            var daviesBouldinIndex = crossValidationResults.Select(r => r.Metrics.DaviesBouldinIndex);
            var normalizedMutualInformation = crossValidationResults.Select(r => r.Metrics.NormalizedMutualInformation);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Clustering model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       Average Distance:                   {averageDistance.Average():0.###} ");
            sb.AppendLine($"*       Average DaviesBouldinIndex:         {daviesBouldinIndex.Average():0.###}  ");
            sb.AppendLine($"*       Average NormalizedMutualInformation: {normalizedMutualInformation.Average():0.###}  ");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            var result = (from fold in crossValidationResults
                          orderby fold.Metrics.AverageDistance descending
                          select fold.Model).First();
            return result;
         }
         /// <summary>
         /// Evaluates scored multiclass classification data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <param name="predictedLabelColumnName">The name of the predicted label column in data.</param>
         /// <param name="topKPredictionCount">
         /// If given a positive value, the Microsoft.ML.Data.MulticlassClassificationMetrics.TopKAccuracy
         /// will be filled with the top-K accuracy, that is, the accuracy assuming we consider
         /// an example with the correct class within the top-K values as being stored "correctly."
         /// </param>
         /// <returns>The evaluation results for these calibrated outputs.</returns>
         public ClusteringMetrics Evaluate(
            IDataView data,
            string labelColumnName = null,
            string scoreColumnName = "Score",
            string featuresColumnName = null)
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.Clustering.Evaluate(data, labelColumnName, scoreColumnName, featuresColumnName);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Clustering model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       AverageDistance:              {metrics.AverageDistance:0.###} ");
            sb.AppendLine($"*       DaviesBouldinIndex:           {metrics.DaviesBouldinIndex:0.###}  ");
            sb.AppendLine($"*       NormalizedMutualInformation:  {metrics.NormalizedMutualInformation:0.###}  ");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            return metrics;
         }
         #endregion
      }
   }

   /// <summary>
   /// Catalogo classificazione multi classe
   /// </summary>
   public partial class ML
   {
      public sealed class MulticlassClassificationCatalog
      {
         #region Fields
         /// <summary>
         /// Owner
         /// </summary>
         private readonly ML ml;
         #endregion
         #region Properties
         /// <summary>
         /// The list of trainers.
         /// </summary>
         public Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers Trainers => ml.Context.MulticlassClassification.Trainers;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="ml">Owner</param>
         public MulticlassClassificationCatalog(ML ml) => this.ml = ml;
         /// <summary>
         /// Run cross-validation over numberOfFolds folds of data, by fitting estimator,
         /// and respecting samplingKeyColumnName if provided. Then evaluate each sub-model
         /// against labelColumnName and return metrics.
         /// </summary>
         /// <param name="data">The data to run cross-validation on.</param>
         /// <param name="estimator">The estimator to fit.</param>
         /// <param name="numberOfFolds">Number of cross-validation folds.</param>
         /// <param name="labelColumnName">The label column (for evaluation).</param>
         /// <param name="samplingKeyColumnName">
         /// Name of a column to use for grouping rows. If two examples share the same value
         /// of the samplingKeyColumnName, they are guaranteed to appear in the same subset
         /// (train or test). This can be used to ensure no label leakage from the train to
         /// the test set. If null no row grouping will be performed.
         /// </param>
         /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
         /// <returns>The best model</returns>
         public ITransformer CrossValidate(
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
            var crossValidationResults = ml.Context.MulticlassClassification.CrossValidate(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
            var metricsInMultipleFolds = crossValidationResults.Select(r => r.Metrics);
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
            var result = (from fold in crossValidationResults
                          orderby fold.Metrics.LogLoss
                          select fold.Model).First();
            return result;
         }
         /// <summary>
         /// Evaluates scored multiclass classification data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <param name="predictedLabelColumnName">The name of the predicted label column in data.</param>
         /// <param name="topKPredictionCount">
         /// If given a positive value, the Microsoft.ML.Data.MulticlassClassificationMetrics.TopKAccuracy
         /// will be filled with the top-K accuracy, that is, the accuracy assuming we consider
         /// an example with the correct class within the top-K values as being stored "correctly."
         /// </param>
         /// <returns>The evaluation results for these calibrated outputs.</returns>
         public MulticlassClassificationMetrics Evaluate(
            IDataView data,
            string labelColumnName = "Label",
            string scoreColumnName = "Score",
            string predictedLabelColumnName = "PredictedLabel",
            int topKPredictionCount = 0)
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.MulticlassClassification.Evaluate(data, labelColumnName, scoreColumnName, predictedLabelColumnName, topKPredictionCount);
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
            return metrics;
         }
         #endregion
      }
   }

   /// <summary>
   /// Catalogo ranking
   /// </summary>
   public partial class ML
   {
      public sealed class RankingCatalog
      {
         #region Fields
         /// <summary>
         /// Owner
         /// </summary>
         private readonly ML ml;
         #endregion
         #region Properties
         /// <summary>
         /// The list of trainers
         /// </summary>
         public Microsoft.ML.RankingCatalog.RankingTrainers Trainers => ml.Context.Ranking.Trainers;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="ml">Owner</param>
         public RankingCatalog(ML ml) => this.ml = ml;
         /// <summary>
         /// Run cross-validation over numberOfFolds folds of data, by fitting estimator,
         /// and respecting samplingKeyColumnName if provided. Then evaluate each sub-model
         /// against labelColumnName and return metrics.
         /// </summary>
         /// <param name="data">The data to run cross-validation on.</param>
         /// <param name="estimator">The estimator to fit.</param>
         /// <param name="numberOfFolds">Number of cross-validation folds.</param>
         /// <param name="labelColumnName">The label column (for evaluation).</param>
         /// <param name="rowGroupColumnName">
         /// The name of the groupId column in data, which is used to group rows. This column
         /// will automatically be used as SamplingKeyColumn when splitting the data for Cross
         /// Validation, as this is required by the ranking algorithms If null no row grouping
         /// will be performed.
         /// </param>
         /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
         /// <returns>The best model</returns>
         public ITransformer CrossValidate(
            IDataView data,
            IEstimator<ITransformer> estimator,
            int numberOfFolds = 5,
            string labelColumnName = "Label",
            string rowGroupColumnName = "GroupId",
            int? seed = null)
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = ml.Context.Ranking.CrossValidate(data, estimator, numberOfFolds, labelColumnName, rowGroupColumnName, seed);
            var discountedCumulativeGains = crossValidationResults.Select(r => DcgScore(r.Metrics.DiscountedCumulativeGains));
            var normalizedDiscountedCumulativeGains = crossValidationResults.Select(r => DcgScore(r.Metrics.NormalizedDiscountedCumulativeGains));
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Ranking model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       Average DiscountedCumulativeGains:           {discountedCumulativeGains.Average():0.###} ");
            sb.AppendLine($"*       Average NormalizedDiscountedCumulativeGains: {normalizedDiscountedCumulativeGains.Average():0.###} ");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            var result = (from fold in crossValidationResults
                          orderby DcgScore(fold.Metrics.NormalizedDiscountedCumulativeGains) descending
                          select fold.Model).First();
            return result;
         }
         /// <summary>
         /// Funzione del calcolo dello score
         /// </summary>
         /// <param name="dcgs">Elenco di discounted cumulative gains</param>
         /// <returns>Lo score</returns>
         private static double DcgScore(IEnumerable<double> dcgs)
         {
            var i = 2.0;
            var result = 0.0;
            foreach (var dcg in dcgs)
               result += dcg / Math.Log(i++);
            return result;
         }
         /// <summary>
         /// Evaluates scored multiclass classification data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="rowGroupColumnName">
         /// The name of the groupId column in data, which is used to group rows. This column
         /// will automatically be used as SamplingKeyColumn when splitting the data for Cross
         /// Validation, as this is required by the ranking algorithms If null no row grouping
         /// will be performed.
         /// </param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <returns>The evaluation results for these calibrated outputs.</returns>
         public RankingMetrics Evaluate(
            IDataView data,
            string labelColumnName = "Label",
            string rowGroupColumnName = "GroupId",
            string scoreColumnName = "Score") =>
            Evaluate(data, null, labelColumnName, rowGroupColumnName, scoreColumnName);
         /// <summary>
         /// Evaluates scored multiclass classification data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="rowGroupColumnName">
         /// The name of the groupId column in data, which is used to group rows. This column
         /// will automatically be used as SamplingKeyColumn when splitting the data for Cross
         /// Validation, as this is required by the ranking algorithms If null no row grouping
         /// will be performed.
         /// </param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <returns>The evaluation results for these calibrated outputs.</returns>
         public RankingMetrics Evaluate(
            IDataView data,
            RankingEvaluatorOptions options,
            string labelColumnName = "Label",
            string rowGroupColumnName = "GroupId",
            string scoreColumnName = "Score")
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
            var metrics =
               options != null ?
               ml.Context.Ranking.Evaluate(data, options, labelColumnName, rowGroupColumnName, scoreColumnName) :
               ml.Context.Ranking.Evaluate(data, labelColumnName, rowGroupColumnName, scoreColumnName);
            var sb = new StringBuilder();
            sb.AppendLine($"*************************************************************************************************************");
            sb.AppendLine($"*       Metrics for Ranking model      ");
            sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
            sb.AppendLine($"*       DCG Score:           {DcgScore(metrics.DiscountedCumulativeGains)}");
            sb.AppendLine($"*       NDCG Score:          {DcgScore(metrics.NormalizedDiscountedCumulativeGains)}");
            sb.AppendLine($"*************************************************************************************************************");
            ml.LogAppend(sb.ToString());
            return metrics;
         }
         #endregion
      }
   }

   /// <summary>
   /// Catalogo regressione
   /// </summary>
   public partial class ML
   {
      public sealed class RegressionCatalog
      {
         #region Fields
         /// <summary>
         /// Owner
         /// </summary>
         private readonly ML ml;
         #endregion
         #region Properties
         /// <summary>
         /// The list of trainers
         /// </summary>
         public Microsoft.ML.RegressionCatalog.RegressionTrainers Trainers => ml.Context.Regression.Trainers;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="ml"></param>
         public RegressionCatalog(ML ml) => this.ml = ml;
         /// <summary>
         /// Run cross-validation over numberOfFolds folds of data, by fitting estimator,
         /// and respecting samplingKeyColumnName if provided. Then evaluate each sub-model
         /// against labelColumnName and return metrics.
         /// </summary>
         /// <param name="data">The data to run cross-validation on.</param>
         /// <param name="estimator">The estimator to fit.</param>
         /// <param name="numberOfFolds">Number of cross-validation folds.</param>
         /// <param name="labelColumnName">The label column (for evaluation).</param>
         /// <param name="samplingKeyColumnName">
         /// Name of a column to use for grouping rows. If two examples share the same value
         /// of the samplingKeyColumnName, they are guaranteed to appear in the same subset
         /// (train or test). This can be used to ensure no label leakage from the train to
         /// the test set. If null no row grouping will be performed.
         /// </param>
         /// <param name="seed">Seed for the random number generator used to select rows for cross-validation folds.</param>
         /// <returns>The best model</returns>
         public ITransformer CrossValidate(
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
            var crossValidationResults = ml.Context.Regression.CrossValidate(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);
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
            var result = (from fold in crossValidationResults
                          orderby fold.Metrics.LossFunction
                          select fold.Model).First();
            return result;
         }
         /// <summary>
         /// Evaluates scored multiclass classification data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <param name="predictedLabelColumnName">The name of the predicted label column in data.</param>
         /// <param name="topKPredictionCount">
         /// If given a positive value, the Microsoft.ML.Data.MulticlassClassificationMetrics.TopKAccuracy
         /// will be filled with the top-K accuracy, that is, the accuracy assuming we consider
         /// an example with the correct class within the top-K values as being stored "correctly."
         /// </param>
         /// <returns>The evaluation results for these calibrated outputs.</returns>
         public RegressionMetrics Evaluate(
            IDataView data,
            string labelColumnName = "Label",
            string scoreColumnName = "Score")
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogAppendLine("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.Regression.Evaluate(data, labelColumnName, scoreColumnName);
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
            return metrics;
         }
         #endregion
      }
   }
}
