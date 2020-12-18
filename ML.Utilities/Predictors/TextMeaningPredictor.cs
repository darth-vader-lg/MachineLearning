using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using ML.Utilities.Data;
using ML.Utilities.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ML.Utilities.Predictors
{
   /// <summary>
   /// Modello per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed class TextMeaningPredictor : Predictor, IDataTextProvider
   {
      #region Fields
      /// <summary>
      /// Dati extra
      /// </summary>
      private DataStorageTextMemory _extraData;
      /// <summary>
      /// Task di commit dei dati
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskCommitData;
      /// <summary>
      /// Task di salvataggio modello
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskSaveModel;
      /// <summary>
      /// Task di training
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskTrain;
      #endregion
      #region Properties
      /// <summary>
      /// Abilitazione al commit automatico dei dati extra di training 
      /// </summary>
      public bool AutoCommitData { get; set; } = true;
      /// <summary>
      /// Abilitazione al salvataggio automatico del modello ogni volta che viene aggiornato 
      /// </summary>
      public bool AutoSaveModel { get; set; } = true;
      /// <summary>
      /// Dati extra
      /// </summary>
      private DataStorageTextMemory ExtraData => _extraData ??= new DataStorageTextMemory();
      /// <summary>
      /// Task di commit dei dati
      /// </summary>
      private CancellableTask TaskCommitData => _taskCommitData ??= new CancellableTask();
      /// <summary>
      /// Task di salvataggio modello
      /// </summary>
      private CancellableTask TaskSaveModel => _taskSaveModel ??= new CancellableTask();
      /// <summary>
      /// Task di training
      /// </summary>
      private CancellableTask TaskTrain => _taskTrain ??= new CancellableTask();
      /// <summary>
      /// Dati
      /// </summary>
      public string TextData
      {
         get => ExtraData.TextData;
         set { OnExtraDataChanged(EventArgs.Empty); ExtraData.TextData = value; }
      }
      #endregion
      #region Events
      /// <summary>
      /// Evento di variazione dati extra di training
      /// </summary>
      public event EventHandler ExtraDataChanged;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public TextMeaningPredictor() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Contesto di machine learning</param>
      public TextMeaningPredictor(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public TextMeaningPredictor(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Aggiunge una linea di dati
      /// </summary>
      /// <param name="data">Linea da aggiungere</param>
      public void AppendData(string data)
      {
         // Aggiunge la linea di dati
         var sb = new StringBuilder();
         sb.Append(TextData);
         sb.AppendLine(data);
         TextData = sb.ToString();
      }
      /// <summary>
      /// Aggiunge una linea di dati
      /// </summary>
      /// <param name="data">Linea da aggiungere</param>
      public void AppendData(params string[] data) => AppendData(FormatDataRow(data));
      /// <summary>
      /// Stoppa il training ed annulla la validita' dei dati
      /// </summary>
      private void CancelTraining()
      {
         // Stoppa il training
         TaskTrain.Cancel();
         // Invalida la valutazione
         SetEvaluation(null);
      }
      /// <summary>
      /// Commit dei dati
      /// </summary>
      public void CommitData() => CommitDataAsync().Wait();
      /// <summary>
      /// Commit asincrono dei dati
      /// </summary>
      /// <returns>Il Task</returns>
      public async Task CommitDataAsync()
      {
         if (string.IsNullOrEmpty(ExtraData.TextData))
            return;
         await TaskCommitData;
         await TaskCommitData.StartNew(async cancel =>
         {
            var data = LoadData(DataStorage, ExtraData);
            await SaveDataAsync(DataStorage, data);
            ExtraData.TextData = null;
         });
      }
      /// <summary>
      /// Formatta una riga di dati da un elenco di dati
      /// </summary>
      /// <param name="data"></param>
      /// <returns></returns>
      public string FormatDataRow(params string[] data)
      {
         // Linea da passare al modello
         var inputLine = new StringBuilder();
         // Quotatura stringhe
         var quote = ((DataStorage as IDataTextOptionsProvider)?.TextOptions?.AllowQuoting ?? true) ? "\"" : "";
         // Separatore di colonne
         var separatorChar = (DataStorage as IDataTextOptionsProvider)?.TextOptions?.Separators?.FirstOrDefault() ?? ',';
         // Loop di costruzione della linea di dati
         var separator = "";
         foreach (var item in data) {
            var text = item ?? "";
            var quoting = quote.Length > 0 && text.TrimStart().StartsWith(quote) && text.TrimEnd().EndsWith(quote) ? "" : quote;
            inputLine.Append($"{separator}{quoting}{text}{quoting}");
            separator = new string(separatorChar, 1);
         }
         return inputLine.ToString();
      }
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init()
      {
         DataStorage = new DataStorageTextMemory();
         ModelStorage = new ModelStorageMemory();
      }
      /// <summary>
      /// Funzione di notifica variazione storage dei dati
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected override void OnDataStorageChanged(EventArgs e)
      {
         try { CancelTraining(); } catch { }
         base.OnDataStorageChanged(e);
      }
      /// <summary>
      /// Funzione di notifica variazione dati extra di training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      private void OnExtraDataChanged(EventArgs e)
      {
         try { CancelTraining(); } catch { }
         try { ExtraDataChanged?.Invoke(this, e); } catch (Exception exc) { Trace.WriteLine(exc); }
      }
      /// <summary>
      /// Funzione di notifica variazione storage del modello
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected override void OnModelStorageChanged(EventArgs e)
      {
         try { CancelTraining(); } catch { }
         base.OnModelStorageChanged(e);
      }
      /// <summary>
      /// Predizione
      /// </summary>
      /// <param name="data">Linea con i dati da usare per la previsione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public string Predict(string data) => PredictAsync(data, CancellationToken.None).Result;
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Elenco di dati da usare per la previsione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public string Predict(IEnumerable<string> data) => PredictAsync(data, CancellationToken.None).Result;
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea con i dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public async Task<string> PredictAsync(string data, CancellationToken cancellation = default)
      {
         // Gestione avvio task di training
         _ = StartTrainingAsync(cancellation);
         // Crea una dataview con i dati di input
         var dataView = LoadData(new DataStorageTextMemory() { TextData = data, TextOptions = (DataStorage as IDataTextOptionsProvider)?.TextOptions ?? new TextLoader.Options() });
         cancellation.ThrowIfCancellationRequested();
         // Attande il modello od un eventuale errore di training
         var evaluation = GetEvaluationAsync();
         await Task.WhenAny(evaluation, TaskTrain.Task).Result;
         cancellation.ThrowIfCancellationRequested();
         // Effettua la predizione
         var prediction = evaluation.Result.Model.Transform(dataView);
         cancellation.ThrowIfCancellationRequested();
         // Restituisce il risultato
         return prediction.GetString("PredictedLabel");
      }
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<string> PredictAsync(IEnumerable<string> data, CancellationToken cancellation = default) => PredictAsync(FormatDataRow(data.ToArray()), cancellation);
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<string> PredictAsync(CancellationToken cancellation = default, params string[] data) => PredictAsync(FormatDataRow(data), cancellation);
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      public void StartTraining() => StartTrainingAsync().Wait();
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      public async override Task StartTrainingAsync(CancellationToken cancellation = default)
      {
         if (TaskTrain.CancellationToken.IsCancellationRequested)
            await StopTrainingAsync(cancellation);
         if (TaskTrain.Task.IsCompleted)
            _ = TaskTrain.StartNew(c => Task.Factory.StartNew(() => Training(c), c, TaskCreationOptions.LongRunning, TaskScheduler.Default), cancellation);
      }
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      public void StopTraining() => StopTrainingAsync().Wait();
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      public async override Task StopTrainingAsync(CancellationToken cancellation = default)
      {
         TaskTrain.Cancel();
         await Task.Run(() => TaskTrain.Task.Wait(cancellation));
      }
      /// <summary>
      /// Routine di training continuo
      /// </summary>
      /// <param name="cancel">Token di cancellazione</param>
      private void Training(CancellationToken cancel)
      {
         try {
            // Segnala lo start del training
            OnTrainingStarted(EventArgs.Empty);
            // Carica il modello
            var model = default(ITransformer);
            var inputSchema = default(DataViewSchema);
            try { model = !string.IsNullOrEmpty(ExtraData.TextData) ? default : LoadModel(out inputSchema); } catch (Exception) { }
            cancel.ThrowIfCancellationRequested();
            // Imposta le opzioni di testo per i dati extra in modo che siano uguali a quelli dello storage principale
            ExtraData.TextOptions = (DataStorage as IDataTextOptionsProvider)?.TextOptions ?? ExtraData.TextOptions;
            // Carica i dati
            var data = LoadData(DataStorage, ExtraData);
            cancel.ThrowIfCancellationRequested();
            // Imposta la valutazione
            SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = data?.Schema ?? inputSchema });
            // Effettua eventuale commit automatico
            if (data != default && !string.IsNullOrEmpty(ExtraData.TextData) && AutoCommitData) {
               try {
                  TaskCommitData.Task.Wait(cancel);
                  if (!cancel.IsCancellationRequested) {
                     TaskCommitData.StartNew(cancel => CommitDataAsync(), cancel).Wait();
                     data = LoadData(DataStorage);
                  }
               }
               catch (Exception exc) { Debug.WriteLine(exc); }
            }
            cancel.ThrowIfCancellationRequested();
            // Trainer
            var trainer = ML.NET.MulticlassClassification.Trainers.SdcaNonCalibrated();
            // Pipe di trasformazione
            var pipe =
               ML.NET.Transforms.Conversion.MapValueToKey("Label").
               Append(ML.NET.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in Evaluation.InputSchema where c.Name != "Label" select c.Name).ToArray())).
               Append(ML.NET.Transforms.CopyColumns("Features", "FeaturizeText")).
               Append(ML.NET.Transforms.NormalizeMinMax("Features")).
               AppendCacheCheckpoint(ML.NET).
               Append(trainer).
               Append(ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // Loop di training continuo
            var prevMetrics = model == default ? default(MulticlassClassificationMetrics) : ML.NET.MulticlassClassification.Evaluate(model.Transform(data));
            var seed = 0;
            var iterationMax = 10;
            var iteration = iterationMax;
            while (!cancel.IsCancellationRequested && --iteration >= 0) {
               try {
                  // Effettua il training
                  var dataView = ML.NET.Data.ShuffleRows(data, seed++);
                  model = pipe.Fit(dataView);
                  var metrics = ML.NET.MulticlassClassification.Evaluate(model.Transform(dataView));
                  //var crossValidation = ml.MulticlassClassification.CrossValidate(dataView, pipe, 5, "Label", null, seed++);
                  //var best = crossValidation.Best();
                  //var model = best.Model;
                  //var metrics = best.Metrics;
                  cancel.ThrowIfCancellationRequested();
                  // Verifica se c'e' un miglioramento; se affermativo salva il nuovo modello
                  if (prevMetrics == default || Evaluation?.Model == default || (metrics.MicroAccuracy >= prevMetrics.MicroAccuracy && metrics.LogLoss < prevMetrics.LogLoss)) {
                     // Emette il log
                     ML.NET.WriteLog("Found best model", $"{nameof(TextMeaningPredictor)}.{nameof(Training)}");
                     ML.NET.WriteLog(metrics.ToText(), $"{nameof(TextMeaningPredictor)}.{nameof(Training)}");
                     ML.NET.WriteLog(metrics.ConfusionMatrix.GetFormattedConfusionTable(), $"{nameof(TextMeaningPredictor)}.{nameof(Training)}");
                     cancel.ThrowIfCancellationRequested();
                     // Eventuale salvataggio automatico modello
                     if (AutoSaveModel) {
                        try {
                           TaskSaveModel.Task.Wait(cancel);
                           if (!cancel.IsCancellationRequested)
                              TaskSaveModel.StartNew(cancel => SaveModelAsync(default, model, Evaluation.InputSchema), cancel);
                        }
                        catch (Exception exc) { Debug.WriteLine(exc); }
                     }
                     prevMetrics = metrics;
                     // Aggiorna la valutazione
                     cancel.ThrowIfCancellationRequested();
                     SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = Evaluation.InputSchema });
                     iteration = iterationMax;
                  }
                  // Ricarica i dati
                  cancel.ThrowIfCancellationRequested();
                  // Effettua eventuale commit automatico
                  if (!string.IsNullOrEmpty(ExtraData.TextData) && AutoCommitData) {
                     try {
                        TaskCommitData.Task.Wait(cancel);
                        if (!cancel.IsCancellationRequested) {
                           TaskCommitData.StartNew(cancel => CommitDataAsync(), cancel).Wait();
                           data = LoadData(DataStorage);
                        }
                     }
                     catch (Exception exc)
                     {
                        Debug.WriteLine(exc);
                        data = LoadData(DataStorage, ExtraData);
                     }
                  }
                  else
                     data = LoadData(DataStorage, ExtraData);
                  cancel.ThrowIfCancellationRequested();
               }
               catch (OperationCanceledException) { }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
                  throw;
               }
            }
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            throw;
         }
         // Segnala la fine del training
         OnTrainingEnded(EventArgs.Empty);
      }
      #endregion
   }
}
