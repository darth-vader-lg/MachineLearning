using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using ML.Utilities.Data;
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
   public sealed class TextMeaning : Predictor, IDataTextProvider
   {
      #region Fields
      /// <summary>
      /// Dati extra
      /// </summary>
      private DataStorageString _extraData;
      /// <summary>
      /// Task di training
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskTrain;
      #endregion
      #region Properties
      /// <summary>
      /// Dati extra
      /// </summary>
      private DataStorageString ExtraData => _extraData ??= new DataStorageString();
      /// <summary>
      /// Task di training
      /// </summary>
      private CancellableTask TaskTrain => _taskTrain ??= new CancellableTask();
      /// <summary>
      /// Dati
      /// </summary>
      public string TextData { get => ExtraData.TextData; set => ExtraData.TextData = value; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public TextMeaning() => DataStorage = new DataStorageString();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Contesto di machine learning</param>
      public TextMeaning(int? seed) : base(seed) => DataStorage = new DataStorageString();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public TextMeaning(MachineLearningContext ml) : base(ml) => DataStorage = new DataStorageString();
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
      public async Task<string> PredictAsync(string data, CancellationToken cancellation)
      {
         // Verifica validita' dello storage di dati
         if (DataStorage is not IDataTextOptionsProvider textOptionsProvider)
            return null;
         // Gestione avvio task di training
         if (TaskTrain.CancellationToken.IsCancellationRequested)
            await TaskTrain;
         if (TaskTrain.Task.IsCompleted)
            _ = TaskTrain.StartNew(cancellation => Task.Factory.StartNew(() => Train(cancellation), cancellation, TaskCreationOptions.LongRunning, TaskScheduler.Default), cancellation);
         // Crea una dataview con i dati di input
         var dataView = LoadData(new DataStorageString() { TextData = data, TextOptions = (DataStorage as IDataTextOptionsProvider)?.TextOptions ?? new TextLoader.Options() });
         cancellation.ThrowIfCancellationRequested();
         // Attande il modello
         var predictor = await GetEvaluation();
         cancellation.ThrowIfCancellationRequested();
         // Effettua la predizione
         var prediction = predictor.Model.Transform(dataView);
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
      public Task<string> PredictAsync(IEnumerable<string> data, CancellationToken cancellation)
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
         // Ritorna il task di predizione asincrona
         return PredictAsync(inputLine.ToString(), cancellation);
      }
      /// <summary>
      /// Routine di training continuo
      /// </summary>
      /// <param name="cancel">Token di cancellazione</param>
      private void Train(CancellationToken cancel)
      {
         // Carica il modello
         var model = LoadModel(out var dataViewSchema);
         // Carica i dati
         ExtraData.TextOptions = (DataStorage as IDataTextOptionsProvider)?.TextOptions ?? ExtraData.TextOptions;
         var data = LoadData(DataStorage, ExtraData);
         // Imposta la valutazione
         SetEvaluation(new Evaluator { Data = data, Model = model, Schema = data?.Schema ?? dataViewSchema });
         // Trainer
         var trainer = ML.NET.MulticlassClassification.Trainers.SdcaNonCalibrated();
         // Pipe di trasformazione
         var pipe =
            ML.NET.Transforms.Conversion.MapValueToKey("Label").
            Append(ML.NET.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in Evaluation.Schema where c.Name != "Label" select c.Name).ToArray())).
            Append(ML.NET.Transforms.CopyColumns("Features", "FeaturizeText")).
            Append(ML.NET.Transforms.NormalizeMinMax("Features")).
            AppendCacheCheckpoint(ML.NET).
            Append(trainer).
            Append(ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
         // Loop di training continuo
         var prevMetrics = default(MulticlassClassificationMetrics);
         var seed = 0;
         var iterationMax = 10;
         var iteration = iterationMax;
         var taskSaveModel = Task.CompletedTask;
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
               if (prevMetrics == default || (metrics.MicroAccuracy >= prevMetrics.MicroAccuracy && metrics.LogLoss < prevMetrics.LogLoss)) {
                  // Emette il log
                  ML.NET.WriteLog("Found best model", $"{nameof(TextMeaning)}.{nameof(Train)}");
                  ML.NET.WriteLog(metrics.ToText(), $"{nameof(TextMeaning)}.{nameof(Train)}");
                  cancel.ThrowIfCancellationRequested();
                  // Salva il modello
                  taskSaveModel.Wait(cancel);
                  cancel.ThrowIfCancellationRequested();
                  taskSaveModel = SaveModelAsync();
                  prevMetrics = metrics;
                  // Aggiorna la valutazione
                  SetEvaluation(new Evaluator { Data = data, Model = model, Schema = Evaluation.Schema });
                  iteration = iterationMax;
               }
               // Ricarica i dati
               data = LoadData(DataStorage, ExtraData);
            }
            catch (OperationCanceledException) { }
            catch (Exception exc) {
               Trace.WriteLine(exc);
               ML.NET.WriteLog(exc.Message, $"{nameof(TextMeaning)}.{nameof(Train)}");
            }
         }
      }
      #endregion
   }
}
