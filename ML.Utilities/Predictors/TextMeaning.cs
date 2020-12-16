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
         var config = new TextLoader.Options
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
         DataStorage = new DataStorageString { Config = config };
      }
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="sentences">Elenco di sentenze da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      public async Task<string> PredictAsync(IEnumerable<string> sentences, CancellationToken cancellation)
      {
         await Task.CompletedTask;
         return "";
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
         while (!cancel.IsCancellationRequested) {
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
               }
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
