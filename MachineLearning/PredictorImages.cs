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
   public sealed partial class PredictorImages : PredictorMulticlass, IDataStorageProvider, IModelStorageProvider, ITextLoaderOptionsProvider
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
      private TextLoader.Options _textLoaderOptions;
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
      public TextLoader.Options TextLoaderOptions
      {
         get
         {
            return _textLoaderOptions ??= new TextLoader.Options
            {
               AllowQuoting = true,
               Separators = new[] { ',' },
               Columns = new[]
               {
                  new TextLoader.Column("Label", DataKind.String, 0),
                  new TextLoader.Column("ImagePath", DataKind.String, 1),
                  new TextLoader.Column("Timestamp", DataKind.DateTime, 2),
               }
            };
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
      public Prediction GetPrediction(string imagePath) => new Prediction(this, GetPredictionData(null, imagePath));
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <param name="cancel">Eventuale token di cancellazione</param>
      /// <returns>Il task di previsione del tipo di immagine</returns>
      public async Task<Prediction> GetPredictionAsync(string imagePath, CancellationToken cancel = default) => new Prediction(this, await GetPredictionDataAsync(cancel, null, imagePath));
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
            Append(Trainers.ImageClassification()).
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
      /// L'elenco puo' essere passato alla funzione di aggiunta di dati di training
      /// </summary>
      /// <param name="path">La directory radice delle immagini</param>
      /// <param name="opt">Opzioni di formattazione del testo di input</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>La lista di dati di training</returns>
      /// <remarks>Le immagini vanno oganizzate in sottodirectory della radice, in cui il nome della sottodirectory specifica la label delle immagini contenute.</remarks>
      public static async Task<string> GetTrainingDataFromPathAsync(string path, TextLoader.Options opt, CancellationToken cancellation = default)
      {
         return await Task.Run(() =>
         {
            var sb = new StringBuilder();
            var dirs = from item in Directory.GetDirectories(path, "*.*", SearchOption.TopDirectoryOnly)
                       where File.GetAttributes(item).HasFlag(FileAttributes.Directory)
                       select item;
            var data = from dir in dirs
                       from file in Directory.GetFiles(dir, "*.*", SearchOption.TopDirectoryOnly)
                       let ext = Path.GetExtension(file).ToLower()
                       where new[] { ".jpg", ".png", ".bmp" }.Contains(ext)
                       let line = $"\"{Path.GetFileName(dir)}\"{opt.Separators[0]}\"{file}\"{opt.Separators[0]}\"{File.GetLastWriteTimeUtc(file):o}\""
                       orderby line
                       select line;
            foreach (var line in data) {
               sb.AppendLine(line);
               cancellation.ThrowIfCancellationRequested();
            }
            return sb.ToString();
         }, cancellation);
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
         var data = await GetTrainingDataFromPathAsync(path, TextLoaderOptions, cancellation);
         cancellation.ThrowIfCancellationRequested();
         ML.NET.WriteLog("Updating image list", Name);
         // Effettua l'aggiornamento dello storage di dati sincronizzandolo con lo stato delle immagini
         await Task.Run(() =>
         {
            // Dati di training
            var trainingData = new DataStorageTextMemory() { TextData = data };
            // Formatta correttamente come fossero salvati
            trainingData.SaveData(this, trainingData.LoadData(this));
            // Legge i dati di storage
            var storageData = new DataStorageTextMemory();
            storageData.SaveData(this, DataStorage.LoadData(this));
            // Confronta e aggiorna lo storage se rilevata variazione
            if (trainingData.TextData != storageData.TextData)
               DataStorage.SaveData(this, trainingData.LoadData(this));
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
         /// <param name="predictor">Previsore</param>
         /// <param name="data">Dati della previsione</param>
         internal Prediction(PredictorImages predictor, IDataView data)
         {
            var grid = data.ToDataViewGrid(predictor);
            Kind = grid[0]["PredictedLabel"];
            var scores = (float[])grid[0]["Score"];
            var slotNames = grid.Schema["Score"].GetSlotNames();
            Scores = slotNames.Zip(scores).Select(item => new KeyValuePair<string, float>(item.First, item.Second)).ToArray();
         }
         #endregion
      }
   }
}
