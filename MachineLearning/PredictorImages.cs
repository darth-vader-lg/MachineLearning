using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Classe per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed partial class PredictorImages : Predictor, IDataStorageProvider, IModelStorageProvider, ITextDataOptionsProvider
   {
      #region Fields
      /// <summary>
      /// Pipe di training
      /// </summary>
      [NonSerialized]
      private IEstimator<ITransformer> _pipe;
      /// <summary>
      /// Contatore di retrain
      /// </summary>
      [NonSerialized]
      private int _retrainCount;
      /// <summary>
      /// Formato dati di training
      /// </summary>
      private TextDataOptions _textOptions;
      #endregion
      #region Properties
      /// <summary>
      /// Abilitazione validazione incrociata
      /// </summary>
      public bool CrossValidation { get; set; }
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Numero massimo di tentativi di retrain del modello
      /// </summary>
      public int MaxRetrain { get; set; } = 1;
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Pipe di training
      /// </summary>
      public IEstimator<ITransformer> Pipe { get => _pipe; set => _pipe = value; }
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      private int Seed { get; set; }
      /// <summary>
      /// Formato dati di training
      /// </summary>
      public TextDataOptions TextDataOptions
      {
         get
         {
            return _textOptions ??= new TextDataOptions(new TextLoader.Options()
            {
               AllowQuoting = true,
               Separators = new[] { ',' },
               Columns = new[]
               {
                  new TextLoader.Column("Label", DataKind.String, 0),
                  new TextLoader.Column("ImagePath", DataKind.String, 1),
                  new TextLoader.Column("Timestamp", DataKind.DateTime, 2),
               }
            });
         }
      }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorImages() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public PredictorImages(int? seed) : base(seed) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorImages(MachineLearningContext ml) : base(ml) { }
      /// <summary>
      /// Funzione di restituzione della migliore fra due valutazioni modello
      /// </summary>
      /// <param name="modelEvaluation1">Prima valutazione</param>
      /// <param name="modelEvaluation2">Seconda valutazione</param>
      /// <returns>La migliore delle due valutazioni</returns>
      /// <remarks>Tenere conto che le valutazioni potrebbero essere null</remarks>
      protected override object GetBestModelEvaluation(object modelEvaluation1, object modelEvaluation2)
      {
         var best = modelEvaluation2;
         if (modelEvaluation1 is MulticlassClassificationMetrics metrics1 && modelEvaluation2 is MulticlassClassificationMetrics metrics2)
            best = metrics2.MicroAccuracy >= metrics1.MicroAccuracy && metrics2.LogLoss < metrics1.LogLoss ? modelEvaluation2 : modelEvaluation1;
         if (best == modelEvaluation2)
            _retrainCount = 0;
         return best;
      }
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <returns>Il tipo di immagine</returns>
      public Prediction GetPrediction(string imagePath) => new Prediction(GetPredictionData(null, imagePath));
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <param name="cancel">Eventuale token di cancellazione</param>
      /// <returns>Il task di previsione del tipo di immagine</returns>
      public async Task<Prediction> GetPredictionAsync(string imagePath, CancellationToken cancel = default) => new Prediction(await GetPredictionDataAsync(cancel, null, imagePath));
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(ITransformer model, IDataView data) => ML.NET.MulticlassClassification.Evaluate(model.Transform(data));
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is not MulticlassClassificationMetrics metrics)
            return null;
         var sb = new StringBuilder();
         sb.AppendLine(metrics.ToText());
         sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
         return sb.ToString();
      }
      /// <summary>
      /// Restituisce il modello effettuando il training. Da implementare nelle classi derivate
      /// </summary>
      /// <param name="dataView">Dati di training</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      protected override ITransformer GetTrainedModel(IDataView dataView, CancellationToken cancellation)
      {
         // Verifica numero di tentativi massimi di retrain raggiunto
         if (++_retrainCount > MaxRetrain)
            return null;
         // Pipe di training
         Pipe ??=
            ML.NET.Transforms.Conversion.MapValueToKey("Label").
            Append(ML.NET.Transforms.LoadRawImageBytes("Features", null, "ImagePath")).
            Append(ML.NET.MulticlassClassification.Trainers.ImageClassification("Label", "Features")).
            Append(ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
         // Training
         if (CrossValidation) {
            var result = ML.NET.MulticlassClassification.CrossValidate(dataView, Pipe, 5, "Label", null, Seed++);
            return result.Best().Model;
         }
         else {
            var data = ML.NET.Data.ShuffleRows(dataView, Seed++);
            return Pipe.Fit(data);
         }
      }
      /// <summary>
      /// Ottiene un elenco di dati di training data la directory radice delle immagini.
      /// </summary>
      /// <param name="path">La directory radice delle immagini</param>
      /// <param name="opt">Opzioni di formattazione del testo di input</param>
      /// <returns>La lista di dati di training</returns>
      /// <remarks>Le immagini vanno oganizzate in sottodirectory della radice, in cui il nome della sottodirectory specifica la label delle immagini contenute.</remarks>
      private static async Task<IEnumerable<string>> GetTrainingDataAsync(string path, TextDataOptions opt)
      {
         return await Task.Run(() =>
         {
            var dirs = from item in Directory.GetDirectories(path, "*.*", SearchOption.TopDirectoryOnly)
                       where File.GetAttributes(item).HasFlag(FileAttributes.Directory)
                       select item;
            var data = from dir in dirs
                       from file in Directory.GetFiles(dir, "*.*", SearchOption.TopDirectoryOnly)
                       let ext = Path.GetExtension(file).ToLower()
                       where new[] { ".jpg", ".png", ".bmp" }.Contains(ext)
                       select $"\"{Path.GetFileName(dir)}\"{opt.Separators[0]}\"{file}\"{opt.Separators[0]}\"{File.GetLastWriteTimeUtc(file):o}\"";
            return data;
         });
      }
      /// <summary>
      /// Ottiene un elenco di dati di training data la directory radice delle immagini.
      /// L'elenco puo' essere passato alla funzione di aggiunta di dati di training
      /// </summary>
      /// <param name="path">La directory radice delle immagini</param>
      /// <param name="opt">Opzioni di formattazione del testo di input</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>La lista di dati di training</returns>
      /// <remarks>Le immagini vanno oganizzate in sottodirectory della radice, in cui il nome della sottodirectory specifica la label delle immagini contenute.</remarks>
      public static async Task<string> GetTrainingDataFromPathAsync(string path, TextDataOptions opt, CancellationToken cancellation = default)
      {
         var sb = new StringBuilder();
         var data = await GetTrainingDataAsync(path, opt);
         await Task.Run(() =>
         {
            foreach (var line in data) {
               sb.AppendLine(line);
               cancellation.ThrowIfCancellationRequested();
            }
         }, cancellation);
         return sb.ToString();
      }
      /// <summary>
      /// Aggiorna lo storage di dati con l'elenco delle immagini categorizzate contenute nella directory indicata
      /// </summary>
      /// <param name="path">La directory radice delle immagini</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <remarks>Le immagini vanno oganizzate in sottodirectory della radice, in cui il nome della sottodirectory specifica la label delle immagini contenute.</remarks>
      public async Task UpdateStorageAsync(string path, CancellationToken cancellation = default)
      {
         // Verifica che sia definito uno storage di dati
         if (DataStorage == null)
            return;
         // Ottiene i dati di training (l'elenco delle immagini classificate e datate)
         var data = await GetTrainingDataAsync(path, TextDataOptions);
         cancellation.ThrowIfCancellationRequested();
         ML.NET.WriteLog("Updating image list", Name);
         // Effettua l'aggiornamento dello storage di dati sincronizzandolo con lo stato delle immagini
         await Task.Run(() =>
         {
            // File temporaneo
            var tmpFile = default(string);
            // Stream temporaneo
            var tmpStream = default(StreamWriter);
            try {
               // Hashset di immagini gia' esistenti nello storage
               var existing = new HashSet<long>();
               // Ottiene un file temporaneo e ne crea lo stream di scrittura
               tmpFile = Path.GetTempFileName();
               tmpStream = new StreamWriter(tmpFile);
               // Flag di abilitazione aggiornamento storage
               var updateDataStorage = false;
               try {
                  // Dati dello storage
                  var storageData = DataStorage.LoadData(ML, TextDataOptions);
                  // Loop su tutte le rige di dati
                  foreach (var dataRow in storageData.ToEnumerable(ML.NET)) {
                     // Flag di abilitazione scrittura linea sullo stram temporaneo
                     var writeLine = true;
                     // Ottiene la linea in formato coppie chiave/valore
                     var dataValues = dataRow.ToKeyValuePairs().First();
                     // Dati della riga
                     var label = Convert.ToString(dataValues.Last(item => item.Key == "Label").Value);
                     var imagePath = Convert.ToString(dataValues.Last(item => item.Key == "ImagePath").Value).ToLower();
                     var timestamp = Convert.ToDateTime(dataValues.Last(item => item.Key == "Timestamp").Value);
                     // Spazzola i dati di training alla ricerca di uno che punti alla stessa immagine
                     var ix = 0L;
                     foreach (var trainingData in data) {
                        // Linea di training
                        var trainingLine = new DataTextMemory(ML, trainingData, TextDataOptions);
                        // Valori di training
                        var trainingValues = trainingLine.LoadData(ML, TextDataOptions).ToKeyValuePairs().First();
                        // Verifica se l'immagine del set di training corrisponde a quella contenuta nel set di dati
                        if (imagePath == Convert.ToString(trainingValues.Last(item => item.Key == "ImagePath").Value).ToLower()) {
                           // Se l'immagine di training e' piu' vecchia o uguale a quella dello storage mantiene quella di storage, altrimenti quella di training
                           if (timestamp >= Convert.ToDateTime(trainingValues.Last(item => item.Key == "Timestamp").Value))
                              trainingLine = new DataTextMemory(ML, dataRow, TextDataOptions);
                           else {
                              ML.NET.WriteLog($"Found updated image: {trainingValues.Last(item => item.Key == "ImagePath").Value}");
                              updateDataStorage = true;
                           }
                           // Scrive la linea scelta nel file temporaneo
                           tmpStream.Write(trainingLine.TextData);
                           existing.Add(ix);
                           writeLine = false;
                        }
                        ix++;
                        cancellation.ThrowIfCancellationRequested();
                     }
                     // Scrive la linea nel file temporaneo se non ci sono state modifiche
                     if (writeLine)
                        tmpStream.Write(new DataTextMemory(ML, dataRow, TextDataOptions));
                  }
               }
               // Il file di storage potrebbe non esistere ancora
               catch (FileNotFoundException) {
               }
               // Aggiunge tutte le linee del set di training che non siano gia' state elaborate nella fase precedente
               var ixTraining = 0L;
               foreach (var trainingData in data) {
                  if (!existing.Contains(ixTraining)) {
                     var formatter = new DataTextMemory(ML, trainingData, TextDataOptions);
                     ML.NET.WriteLog($"Found new image: {formatter.LoadData(ML, TextDataOptions).GetString("ImagePath")}");
                     tmpStream.Write(formatter.TextData);
                     updateDataStorage = true;
                  }
                  ixTraining++;
                  cancellation.ThrowIfCancellationRequested();
               }
               // Chiude il file temporaneo
               tmpStream.Close();
               tmpStream = null;
               // Aggiorna lo storage con il contenuto del file temporaneo se necessario
               if (updateDataStorage) {
                  cancellation.ThrowIfCancellationRequested();
                  DataStorage.SaveData(ML, new DataTextFile(tmpFile).LoadData(ML, TextDataOptions), TextDataOptions, SaveDataSchemaComment);
               }
            }
            finally {
               // Chiude lo stream temporaneo
               try {
                  if (tmpStream != null)
                     tmpStream.Close();
               }
               catch (Exception) { }
               // Cancella il file temporaneo
               try {
                  if (File.Exists(tmpFile))
                     File.Delete(tmpFile);
               }
               catch (Exception) { }
            }
         }, cancellation);
      }
      #endregion
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class PredictorImages // Prediction
   {
      [Serializable]
      public class Prediction
      {
         #region Properties
         /// <summary>
         /// Significato
         /// </summary>
         public string Kind { get; }
         /// <summary>
         /// Punteggio per il tipo previsto
         /// </summary>
         public float Score { get; }
         /// <summary>
         /// Punteggi per label
         /// </summary>
         public KeyValuePair<string, float>[] Scores { get; }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="data">Dati della previsione</param>
         internal Prediction(IDataView data)
         {
            Kind = data.GetString("PredictedLabel");
            var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
            data.Schema["Score"].GetSlotNames(ref slotNames);
            var scores = data.GetValue<VBuffer<float>>("Score");
            Scores = slotNames.GetValues().ToArray().Zip(scores.GetValues().ToArray()).Select(item => new KeyValuePair<string, float>(item.First.ToString(), item.Second)).ToArray();
            Score = Scores.FirstOrDefault(s => s.Key == Kind).Value;
         }
         #endregion
      }
   }
}
