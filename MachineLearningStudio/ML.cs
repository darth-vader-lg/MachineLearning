using Microsoft.ML;
using Microsoft.ML.AutoML;
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
   public partial class ML :
      IChannelProvider,
      IExceptionContext,
      IHostEnvironment,
      IProgress<RunDetail<BinaryClassificationMetrics>>,
      IProgress<RunDetail<MulticlassClassificationMetrics>>,
      IProgress<RunDetail<RankingMetrics>>,
      IProgress<RunDetail<RegressionMetrics>>,
      IProgressChannelProvider
   {
      #region Fields
      /// <summary>
      /// Buffer di log
      /// </summary>
      private readonly StringBuilder logBuilder = new StringBuilder();
      #endregion
      #region Properties
      /// <summary>
      /// Trainers e tasks specifici per i broblemi di rilevamento anomalie.
      /// </summary>
      public AnomalyDetectionCatalog AnomalyDetection { get; }
      /// <summary>
      /// Catalogo di tutte le possibili operazioni di valutazione automatica del miglior algoritmo di training
      /// </summary>
      public AutoCatalog Auto => Context.Auto();
      /// <summary>
      /// Trainers e tasks specifici dei problemi di classificazione binaria.
      /// </summary>
      public BinaryClassificationCatalog BinaryClassification { get; }
      /// <summary>
      /// Trainers e tasks specifici dei problemi di clusterizzazione.
      /// </summary>
      public ClusteringCatalog Clustering { get; }
      /// <summary>
      /// Catalogo di componenti che sara' usato per il caricamento modelli
      /// </summary>
      public ComponentCatalog ComponentCatalog => Context.ComponentCatalog;
      /// <summary>
      /// Log completo
      /// </summary>
      public string CompleteLog { get { lock (logBuilder) return logBuilder.ToString(); } }
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      public MLContext Context { get; }
      /// <summary>
      /// Caricamento e salvataggio dati
      /// </summary>
      public DataOperationsCatalog Data => Context.Data;
      /// <summary>
      /// Trainers e tasks specifici per problemi di previsione.
      /// </summary>
      public ForecastingCatalog Forecasting => Context.Forecasting;
      /// <summary>
      /// Una stringa descrivente il contesto stesso
      /// </summary>
      string IExceptionContext.ContextDescription => ((IExceptionContext)Context).ContextDescription;
      /// <summary>
      /// Operazioni con i modelli di training
      /// </summary>
      public ModelOperationsCatalog Model => Context.Model;
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
      /// <summary>
      /// Operazioni per il processo dei dati
      /// </summary>
      public TransformsCatalog Transforms => Context.Transforms;
      #endregion
      #region Events and delegates
      /// <summary>
      /// Evento messaggio di log
      /// </summary>
      public event EventHandler<LoggingEventArgs> Log;
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
         AnomalyDetection = new AnomalyDetectionCatalog(this);
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
            lock (logBuilder)
               logBuilder.Append(e.Message + Environment.NewLine);
            OnLog(e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Avvia un canale standard di messaggi
      /// </summary>
      IChannel IChannelProvider.Start(string name) => ((IChannelProvider)Context).Start(name);
      /// <summary>
      /// Avvia una pipe di informazione generica
      /// </summary>
      IPipe<TMessage> IChannelProvider.StartPipe<TMessage>(string name) => ((IChannelProvider)Context).StartPipe<TMessage>(name);
      /// <summary>
      /// Processa un eccezione
      /// </summary>
      TException IExceptionContext.Process<TException>(TException ex) => ((IExceptionContext)Context).Process(ex);
      /// <summary>
      /// Crea un host col il nome di registrazione fornito
      /// </summary>
      IHost IHostEnvironment.Register(string name, int? seed, bool? verbose) => ((IHostEnvironment)Context).Register(name, seed, verbose);
      /// <summary>
      /// Progress del machine learning automatico della categoria di classificazione binaria
      /// </summary>
      /// <param name="value">Dati del progress</param>
      void IProgress<RunDetail<BinaryClassificationMetrics>>.Report(RunDetail<BinaryClassificationMetrics> value) => Report(value, value.Exception);
      /// <summary>
      /// Progress del machine learning automatico della categoria di classificazione multiclasse
      /// </summary>
      /// <param name="value">Dati del progress</param>
      void IProgress<RunDetail<MulticlassClassificationMetrics>>.Report(RunDetail<MulticlassClassificationMetrics> value) => Report(value, value.Exception);
      /// <summary>
      /// Progress del machine learning automatico della categoria di ranking
      /// </summary>
      /// <param name="value">Dati del progress</param>
      void IProgress<RunDetail<RankingMetrics>>.Report(RunDetail<RankingMetrics> value) => Report(value, value.Exception);
      /// <summary>
      /// Progress del machine learning automatico della categoria di regressione
      /// </summary>
      /// <param name="value">Dati del progress</param>
      void IProgress<RunDetail<RegressionMetrics>>.Report(RunDetail<RegressionMetrics> value) => Report(value, value.Exception);
      /// <summary>
      /// Riporta un messaggio di log di AutoML
      /// </summary>
      /// <param name="runDetail"></param>
      /// <param name="exception"></param>
      private void Report(RunDetail runDetail, Exception exception = null)
      {
         try {
            if (exception == null)
               Context_Log(this, new LoggingEventArgs($"Trainer: {runDetail.TrainerName}\t{runDetail.RuntimeInSeconds:0.#} secs", ChannelMessageKind.Info, "AutoML"));
            else
               Context_Log(this, new LoggingEventArgs($"Exception: {exception.Message}", ChannelMessageKind.Error, "AutoML"));
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Avvia un canale di progress per un nuome di una computazione
      /// </summary>
      IProgressChannel IProgressChannelProvider.StartProgressChannel(string name) => ((IProgressChannelProvider)Context).StartProgressChannel(name);
      /// <summary>
      /// Carica un modello
      /// </summary>
      /// <param name="path">Path del modello</param>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(string path, out DataViewSchema inputSchema)
      {
         LogMessage($"============== Loading the model  ===============");
         var result = Context.Model.Load(path, out inputSchema);
         LogMessage($"The model is loaded from {path}");
         return result;
      }
      /// <summary>
      /// Funzione di aggiunta messaggio al log
      /// </summary>
      /// <param name="text">Testo da stampare</param>
      /// <param name="kind">Tipo di messaggio</param>
      /// <param name="source">Sorgente del messaggio</param>
      public void LogMessage(string text, ChannelMessageKind kind = ChannelMessageKind.Info, string source = null)
      {
         try {
            if (string.IsNullOrEmpty(text))
               return;
            lock (logBuilder)
               logBuilder.AppendLine(text);
            OnLog(new LoggingEventArgs(text, kind, source ?? $"{nameof(ML)}.{nameof(LogMessage)}"));
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di logging dei messaggi
      /// </summary>
      /// <param name="e">Argomenti del log</param>
      protected virtual void OnLog(LoggingEventArgs e)
      {
         try {
            Log?.Invoke(this, e);
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
         LogMessage($"=============== Saving the model  ===============");
         Context.Model.Save(model, schema, path);
         LogMessage($"The model is saved to {path}");
      }
      /// <summary>
      /// Funzione di training del modello
      /// </summary>
      /// <param name="trainingDataView">Dati di training</param>
      /// <param name="trainingPipeline">Pipeline di training</param>
      /// <returns></returns>
      public ITransformer TrainModel(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
      {
         LogMessage("=============== Training  model ===============");
         var model = trainingPipeline.Fit(trainingDataView);
         LogMessage("=============== End of training process ===============");
         return model;
      }
      #endregion
   }

   /// <summary>
   /// Catalogo rilevazione anomalie
   /// </summary>
   public partial class ML
   {
      public sealed class AnomalyDetectionCatalog
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
         public Microsoft.ML.AnomalyDetectionCatalog.AnomalyDetectionTrainers Trainers => ml.Context.AnomalyDetection.Trainers;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="ml">Owner</param>
         public AnomalyDetectionCatalog(ML ml) => this.ml = ml;
         /// <summary>
         /// Method to modify the threshold to existing model and return modified model.
         /// </summary>
         /// <typeparam name="TModel">The type of the model parameters.</typeparam>
         /// <param name="model">Existing model to modify threshold.</param>
         /// <param name="threshold">New threshold.</param>
         /// <returns>New model with modified threshold.</returns>
         public AnomalyPredictionTransformer<TModel> ChangeModelThreshold<TModel>(AnomalyPredictionTransformer<TModel> model, float threshold) where TModel : class =>
            ml.Context.AnomalyDetection.ChangeModelThreshold(model, threshold);
         /// <summary>
         /// Evaluates scored anomaly detection data.
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
         /// <param name="predictedLabelColumnName">The name of the predicted label column in data.</param>
         /// <param name="falsePositiveCount">The number of false positives to compute the Microsoft.ML.Data.AnomalyDetectionMetrics.DetectionRateAtFalsePositiveCount metric.</param>
         /// <returns>Evaluation results.</returns>
         public AnomalyDetectionMetrics Evaluate(
            IDataView data,
            string labelColumnName = "Label",
            string scoreColumnName = "Score",
            string predictedLabelColumnName = "PredictedLabel",
            int falsePositiveCount = 10)
         {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            ml.LogMessage("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.AnomalyDetection.Evaluate(data, labelColumnName, scoreColumnName, predictedLabelColumnName, falsePositiveCount);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Binary classification model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       AreaUnderRocCurve:                  {metrics.AreaUnderRocCurve:0.###} ");
            ml.LogMessage($"*       DetectionRateAtFalsePositiveCount:  {metrics.DetectionRateAtFalsePositiveCount:0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
            return metrics;
         }
         #endregion
      }
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
            ml.LogMessage("=============== Cross-validating to get model's accuracy metrics ===============");
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
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Binary classification model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Average Accuracy:                      {accuracy.Average():0.###} ");
            ml.LogMessage($"*       Average AreaUnderPrecisionRecallCurve: {areaUnderPrecisionRecallCurve.Average():0.###}  ");
            ml.LogMessage($"*       Average AreaUnderRocCurve:             {areaUnderRocCurve.Average():0.###}  ");
            ml.LogMessage($"*       Average Entropy:                       {entropy.Average():0.###}  ");
            ml.LogMessage($"*       Average F1Score:                       {F1Score.Average():0.###}  ");
            ml.LogMessage($"*       Average LogLoss:                       {logLoss.Average():0.###}  ");
            ml.LogMessage($"*       Average LogLossReduction:              {logLossReduction.Average():0.###}  ");
            ml.LogMessage($"*       Average NegativePrecision:             {negativePrecision.Average():0.###}  ");
            ml.LogMessage($"*       Average NegativeRecall:                {negativeRecall.Average():0.###}  ");
            ml.LogMessage($"*       Average PositivePrecision:             {positivePrecision.Average():0.###}  ");
            ml.LogMessage($"*       Average PositiveRecall:                {positiveRecall.Average():0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = ml.Context.BinaryClassification.CrossValidateNonCalibrated(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
            var accuracy = crossValidationResults.Select(r => r.Metrics.Accuracy);
            var areaUnderPrecisionRecallCurve = crossValidationResults.Select(r => r.Metrics.AreaUnderPrecisionRecallCurve);
            var areaUnderRocCurve = crossValidationResults.Select(r => r.Metrics.AreaUnderRocCurve);
            var F1Score = crossValidationResults.Select(r => r.Metrics.F1Score);
            var negativePrecision = crossValidationResults.Select(r => r.Metrics.NegativePrecision);
            var negativeRecall = crossValidationResults.Select(r => r.Metrics.NegativeRecall);
            var positivePrecision = crossValidationResults.Select(r => r.Metrics.PositivePrecision);
            var positiveRecall = crossValidationResults.Select(r => r.Metrics.PositiveRecall);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Binary classification model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Average Accuracy:                      {accuracy.Average():0.###} ");
            ml.LogMessage($"*       Average AreaUnderPrecisionRecallCurve: {areaUnderPrecisionRecallCurve.Average():0.###}  ");
            ml.LogMessage($"*       Average AreaUnderRocCurve:             {areaUnderRocCurve.Average():0.###}  ");
            ml.LogMessage($"*       Average F1Score:                       {F1Score.Average():0.###}  ");
            ml.LogMessage($"*       Average NegativePrecision:             {negativePrecision.Average():0.###}  ");
            ml.LogMessage($"*       Average NegativeRecall:                {negativeRecall.Average():0.###}  ");
            ml.LogMessage($"*       Average PositivePrecision:             {positivePrecision.Average():0.###}  ");
            ml.LogMessage($"*       Average PositiveRecall:                {positiveRecall.Average():0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.BinaryClassification.Evaluate(data, labelColumnName, scoreColumnName, probabilityColumnName, predictedLabelColumnName);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Binary classification model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Accuracy:                     {metrics.Accuracy:0.###} ");
            ml.LogMessage($"*       AreaUnderPrecisionRecallCurve:{metrics.AreaUnderPrecisionRecallCurve:0.###}  ");
            ml.LogMessage($"*       AreaUnderRocCurve:            {metrics.AreaUnderRocCurve:0.###}  ");
            ml.LogMessage($"*       Entropy:                      {metrics.Entropy:0.###}  ");
            ml.LogMessage($"*       F1Score:                      {metrics.F1Score:0.###}  ");
            ml.LogMessage($"*       LogLoss:                      {metrics.LogLoss:0.###}  ");
            ml.LogMessage($"*       LogLossReduction:             {metrics.LogLossReduction:0.###}  ");
            ml.LogMessage($"*       NegativePrecision:            {metrics.NegativePrecision:0.###}  ");
            ml.LogMessage($"*       NegativeRecall:               {metrics.NegativeRecall:0.###}  ");
            ml.LogMessage($"*       PositivePrecision:            {metrics.PositivePrecision:0.###}  ");
            ml.LogMessage($"*       PositiveRecall:               {metrics.PositiveRecall:0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.BinaryClassification.EvaluateNonCalibrated(data, labelColumnName, scoreColumnName, predictedLabelColumnName);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Binary classification model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Accuracy:                     {metrics.Accuracy:0.###} ");
            ml.LogMessage($"*       AreaUnderPrecisionRecallCurve:{metrics.AreaUnderPrecisionRecallCurve:0.###}  ");
            ml.LogMessage($"*       AreaUnderRocCurve:            {metrics.AreaUnderRocCurve:0.###}  ");
            ml.LogMessage($"*       F1Score:                      {metrics.F1Score:0.###}  ");
            ml.LogMessage($"*       NegativePrecision:            {metrics.NegativePrecision:0.###}  ");
            ml.LogMessage($"*       NegativeRecall:               {metrics.NegativeRecall:0.###}  ");
            ml.LogMessage($"*       PositivePrecision:            {metrics.PositivePrecision:0.###}  ");
            ml.LogMessage($"*       PositiveRecall:               {metrics.PositiveRecall:0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = ml.Context.Clustering.CrossValidate(data, estimator, numberOfFolds, labelColumnName, featuresColumnName, samplingKeyColumnName, seed);
            var averageDistance = crossValidationResults.Select(r => r.Metrics.AverageDistance);
            var daviesBouldinIndex = crossValidationResults.Select(r => r.Metrics.DaviesBouldinIndex);
            var normalizedMutualInformation = crossValidationResults.Select(r => r.Metrics.NormalizedMutualInformation);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Clustering model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Average Distance:                   {averageDistance.Average():0.###} ");
            ml.LogMessage($"*       Average DaviesBouldinIndex:         {daviesBouldinIndex.Average():0.###}  ");
            ml.LogMessage($"*       Average NormalizedMutualInformation: {normalizedMutualInformation.Average():0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
            var result = (from fold in crossValidationResults
                          orderby fold.Metrics.AverageDistance descending
                          select fold.Model).First();
            return result;
         }
         /// <summary>
         /// Evaluates scored clustering
         /// </summary>
         /// <param name="data">The scored data.</param>
         /// <param name="labelColumnName">The name of the label column in data.</param>
         /// <param name="scoreColumnName">The name of the score column in data.</param>
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
            ml.LogMessage("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.Clustering.Evaluate(data, labelColumnName, scoreColumnName, featuresColumnName);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Clustering model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       AverageDistance:              {metrics.AverageDistance:0.###} ");
            ml.LogMessage($"*       DaviesBouldinIndex:           {metrics.DaviesBouldinIndex:0.###}  ");
            ml.LogMessage($"*       NormalizedMutualInformation:  {metrics.NormalizedMutualInformation:0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("=============== Cross-validating to get model's accuracy metrics ===============");
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
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Multi-class Classification model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:0.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:0.###})");
            ml.LogMessage($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:0.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:0.###})");
            ml.LogMessage($"*       Average LogLoss:          {logLossAverage:0.###}  - Standard deviation: ({logLossStdDeviation:0.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:0.###})");
            ml.LogMessage($"*       Average LogLossReduction: {logLossReductionAverage:0.###}  - Standard deviation: ({logLossReductionStdDeviation:0.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:0.###})");
            var result = (from item in (from cvr in crossValidationResults
                                        group cvr by cvr.Metrics.MicroAccuracy into grps
                                        orderby grps.Key descending
                                        select grps).First()
                          orderby item.Metrics.LogLoss
                          select item).First();
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       MicroAccuracy:    {result.Metrics.MicroAccuracy:0.###}");
            ml.LogMessage($"*       MacroAccuracy:    {result.Metrics.MacroAccuracy:0.###}");
            ml.LogMessage($"*       LogLoss:          {result.Metrics.LogLoss:0.###}");
            ml.LogMessage($"*       LogLossReduction: {result.Metrics.LogLossReduction:0.###}");
            ml.LogMessage($"*************************************************************************************************************");
            return result.Model;
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
            ml.LogMessage("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.MulticlassClassification.Evaluate(data, labelColumnName, scoreColumnName, predictedLabelColumnName, topKPredictionCount);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Multi-class Classification model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            ml.LogMessage($"*       MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            ml.LogMessage($"*       LogLoss:          {metrics.LogLoss:0.###}");
            ml.LogMessage($"*       LogLossReduction: {metrics.LogLossReduction:0.###}");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = ml.Context.Ranking.CrossValidate(data, estimator, numberOfFolds, labelColumnName, rowGroupColumnName, seed);
            var discountedCumulativeGains = crossValidationResults.Select(r => DcgScore(r.Metrics.DiscountedCumulativeGains));
            var normalizedDiscountedCumulativeGains = crossValidationResults.Select(r => DcgScore(r.Metrics.NormalizedDiscountedCumulativeGains));
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Ranking model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Average DiscountedCumulativeGains:           {discountedCumulativeGains.Average():0.###} ");
            ml.LogMessage($"*       Average NormalizedDiscountedCumulativeGains: {normalizedDiscountedCumulativeGains.Average():0.###} ");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("================== Evaluating to get model's accuracy metrics ==================");
            var metrics =
               options != null ?
               ml.Context.Ranking.Evaluate(data, options, labelColumnName, rowGroupColumnName, scoreColumnName) :
               ml.Context.Ranking.Evaluate(data, labelColumnName, rowGroupColumnName, scoreColumnName);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Ranking model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       DCG Score:           {DcgScore(metrics.DiscountedCumulativeGains)}");
            ml.LogMessage($"*       NDCG Score:          {DcgScore(metrics.NormalizedDiscountedCumulativeGains)}");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = ml.Context.Regression.CrossValidate(data, estimator, numberOfFolds, labelColumnName, samplingKeyColumnName, seed);
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Regression model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       Average L1 Loss:       {L1.Average():0.###} ");
            ml.LogMessage($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            ml.LogMessage($"*       Average RMS:           {RMS.Average():0.###}  ");
            ml.LogMessage($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            ml.LogMessage($"*       Average R-squared:     {R2.Average():0.###}  ");
            ml.LogMessage($"*************************************************************************************************************");
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
            ml.LogMessage("================== Evaluating to get model's accuracy metrics ==================");
            var metrics = ml.Context.Regression.Evaluate(data, labelColumnName, scoreColumnName);
            ml.LogMessage($"*************************************************************************************************************");
            ml.LogMessage($"*       Metrics for Regression model      ");
            ml.LogMessage($"*------------------------------------------------------------------------------------------------------------");
            ml.LogMessage($"*       LossFunction:        {metrics.LossFunction:0.###}");
            ml.LogMessage($"*       MeanAbsoluteError:   {metrics.MeanAbsoluteError:0.###}");
            ml.LogMessage($"*       MeanSquaredError:    {metrics.MeanSquaredError:0.###}");
            ml.LogMessage($"*       RootMeanSquaredError:{metrics.RootMeanSquaredError:0.###}");
            ml.LogMessage($"*       RSquared:            {metrics.RSquared:0.###}");
            ml.LogMessage($"*************************************************************************************************************");
            return metrics;
         }
         #endregion
      }
   }
}
