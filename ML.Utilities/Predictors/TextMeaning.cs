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
      /// Aggiunge una linea
      /// </summary>
      /// <param name="line">Linea da aggiungere</param>
      public void AppendLine(string line)
      {
         var sb = new StringBuilder();
         sb.Append(TextData);
         sb.AppendLine(line);
         TextData = sb.ToString();
      }
      /// <summary>
      /// Inizializzazione
      /// </summary>
      /// <param name="columns">Elenco dei nomi di colonne</param>
      /// <param name="labelColumnName">Nome della colonna label</param>
      private void Init(IEnumerable<string> columns = null, string labelColumnName = "Label")
      {
         this.labelColumnName = string.IsNullOrWhiteSpace(labelColumnName) ? "Label" : labelColumnName;
         var textOptions = new TextLoader.Options
         {
            AllowQuoting = true,
            AllowSparse = false,
            Separators = new[] { ',' },
            Columns = columns != default ?
            columns.Select((c, i) => new TextLoader.Column(c != this.labelColumnName ? c : labelColumnName, DataKind.String, i)).ToArray() :
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
      /// <param name="sentences">Linea con le sentenze da usare per la previsione</param>
      /// <returns>Il task di predizione</returns>
      public string Predict(string sentences) => PredictAsync(sentences, CancellationToken.None).Result;
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="sentences">Elenco di sentenze da usare per la previsione</param>
      /// <returns>Il task di predizione</returns>
      public string Predict(IEnumerable<string> sentences) => PredictAsync(sentences, CancellationToken.None).Result;
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="sentences">Linea con le sentenze da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      public async Task<string> PredictAsync(string sentences, CancellationToken cancellation)
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
         var dataView = LoadData(new DataStorageString() { TextData = sentences, TextOptions = textOptionsProvider.TextOptions });
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
      /// <param name="sentences">Elenco di sentenze da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      public Task<string> PredictAsync(IEnumerable<string> sentences, CancellationToken cancellation)
      {
         // Linea da passare al modello
         string inputLine;
         // Costruzione in presenza di opzioni di testo
         if (DataStorage is ITextOptionsProvider textOptionsProvider) {
            var textOptions = textOptionsProvider.TextOptions;
            var quote = textOptions.AllowQuoting ? "\"" : "";
            var sb = new StringBuilder();
            var sentencesEnumerator = sentences.GetEnumerator();
            var separator = "";
            for (var i = 0; i < textOptions.Columns.Length; i++) {
               var c = textOptions.Columns[i];
               if (c.Name == labelColumnName)
                  sb.Append(separator);
               else {
                  var sentence = sentencesEnumerator.MoveNext() ? sentencesEnumerator.Current : "";
                  var quoting = sentence.Trim().StartsWith(quote) ? "" : quote;
                  sb.Append($"{separator}{quoting}{sentence}{quoting}");
               }
               separator = new string(new[] { textOptions.Separators[0] });
               cancellation.ThrowIfCancellationRequested();
            }
            inputLine = sb.ToString();
         }
         // Costruzione senza opzioni di testo
         else {
            var separator = "";
            var sb = new StringBuilder();
            var quote = "\"";
            foreach (var sentence in sentences) {
               var quoting = sentence.Trim().StartsWith(quote) ? "" : quote;
               sb.Append($"{separator}{quoting}{sentence}{quoting}");
               separator = ",";
               cancellation.ThrowIfCancellationRequested();
            }
            inputLine = sb.ToString();
         }
         return PredictAsync(inputLine, cancellation);
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
            MLContext.Transforms.Conversion.MapValueToKey("Label").
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
                  MLContext.WriteLog("Found best model", nameof(Train));
                  MLContext.WriteLog(metrics.ToText(), nameof(Train));
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
               MLContext.WriteLog(exc.Message, nameof(Train));
            }
         }
      }
      #endregion
   }
}
