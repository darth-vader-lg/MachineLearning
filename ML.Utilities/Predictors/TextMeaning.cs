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
   public class TextMeaning : Predictor, IDataTextProvider
   {
      #region Fields
      /// <summary>
      /// Nome della colonna label
      /// </summary>
      private string labelColumnName = "Label";
      /// <summary>
      /// Task di training
      /// </summary>
      private (Task Task, CancellationTokenSource Canc) taskTrain = (Task.CompletedTask, new CancellationTokenSource());
      #endregion
      #region Properties
      /// <summary>
      /// Dati
      /// </summary>
      public string TextData { get => (DataStorage as IDataTextProvider)?.TextData; set => (DataStorage as IDataTextProvider).TextData = value; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public TextMeaning() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Contesto di machine learning</param>
      public TextMeaning(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public TextMeaning(MLContext ml) : base(ml) => Init();
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
      /// Inizializzazione
      /// </summary>
      /// <param name="columns">Elenco dei nomi di colonne</param>
      /// <param name="labelColumnName">Nome della colonna label</param>
      private void Init(IEnumerable<string> columns = null, string labelColumnName = "Label")
      {
         // Nome colonna di label
         this.labelColumnName = string.IsNullOrWhiteSpace(labelColumnName) ? "Label" : labelColumnName;
         var textOptions = new TextLoader.Options
         {
            AllowQuoting = true,
            AllowSparse = false,
            Separators = new[] { ',' },
            Columns = columns != default ?
            columns.Select((c, i) => new TextLoader.Column(c, DataKind.String, i)).ToArray() :
            new[]
            {
               new TextLoader.Column("Label", DataKind.String, 0),
               new TextLoader.Column("Sentence", DataKind.String, 1),
            }
         };
         DataStorage = new DataStorageString { TextOptions = textOptions };
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
         if (DataStorage is not ITextOptionsProvider textOptionsProvider)
            return null;
         // Gestione avvio task di training
         if (taskTrain.Canc.IsCancellationRequested)
            await taskTrain.Task;
         if (taskTrain.Task.IsCompleted) {
            taskTrain.Canc = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            taskTrain.Task = Task.Factory.StartNew(() => Train(taskTrain.Canc.Token), taskTrain.Canc.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);
         }
         // Crea una dataview con i dati di input
         var dataView = LoadData(new DataStorageString() { TextData = data, TextOptions = (DataStorage as ITextOptionsProvider)?.TextOptions ?? new TextLoader.Options() });
         cancellation.ThrowIfCancellationRequested();
         // Attande il modello
         var predictor = await TaskModelEvaluation;
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
         var quote = ((DataStorage as ITextOptionsProvider)?.TextOptions?.AllowQuoting ?? true) ? "\"" : "";
         // Separatore di colonne
         var separatorChar = (DataStorage as ITextOptionsProvider)?.TextOptions?.Separators?.FirstOrDefault() ?? ',';
         // Loop di costruzione della linea di dati
         var separator = "";
         foreach (var item in data) {
            var quoting = quote.Length > 0 && item.TrimStart().StartsWith(quote) && item.TrimEnd().EndsWith(quote) ? "" : quote;
            inputLine.Append($"{separator}{quoting}{item}{quoting}");
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
         LoadModel();
         // Carica i dati
         var dataView = LoadData();
         // Trainer
         var trainer = MLContext.MulticlassClassification.Trainers.SdcaNonCalibrated();
         // Pipe di trasformazione
         var pipe =
            MLContext.Transforms.Conversion.MapValueToKey("Label", labelColumnName).
            Append(MLContext.Transforms.Text.FeaturizeText("Sentence_tf", new TextFeaturizingEstimator.Options(), (from c in dataView.Schema where c.Name != "Label" select c.Name).ToArray())).
            Append(MLContext.Transforms.CopyColumns("Features", "Sentence_tf")).
            Append(MLContext.Transforms.NormalizeMinMax("Features")).
            AppendCacheCheckpoint(MLContext).
            Append(trainer).
            Append(MLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
         // Loop di training continuo
         var prevMetrics = default(MulticlassClassificationMetrics);
         var seed = 0;
         var iterationMax = 10;
         var iteration = iterationMax;
         while (!cancel.IsCancellationRequested && --iteration >= 0) {
            try {
               // Effettua il training
               dataView = MLContext.Data.ShuffleRows(dataView, seed++);
               var model = pipe.Fit(dataView);
               var metrics = MLContext.MulticlassClassification.Evaluate(model.Transform(dataView));
               //var crossValidation = ml.MulticlassClassification.CrossValidate(dataView, pipe, 5, "Label", null, seed++);
               //var best = crossValidation.Best();
               //var model = best.Model;
               //var metrics = best.Metrics;
               cancel.ThrowIfCancellationRequested();
               // Verifica se c'e' un miglioramento; se affermativo salva il nuovo modello
               if (prevMetrics == default || (metrics.MicroAccuracy >= prevMetrics.MicroAccuracy && metrics.LogLoss < prevMetrics.LogLoss)) {
                  // Emette il log
                  MLContext.WriteLog("Found best model", $"{nameof(TextMeaning)}.{nameof(Train)}");
                  MLContext.WriteLog(metrics.ToText(), $"{nameof(TextMeaning)}.{nameof(Train)}");
                  cancel.ThrowIfCancellationRequested();
                  // Salva il modello
                  SaveModel();
                  prevMetrics = metrics;
                  // Aggiorna il modello attuale
                  SetModel(model);
                  iteration = iterationMax;
               }
               // Ricarica i dati
               dataView = LoadData();
            }
            catch (OperationCanceledException) { }
            catch (Exception exc) {
               Trace.WriteLine(exc);
               MLContext.WriteLog(exc.Message, $"{nameof(TextMeaning)}.{nameof(Train)}");
            }
         }
      }
      #endregion
   }
}
