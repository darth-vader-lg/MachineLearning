﻿using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Classe per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed partial class ImageRecognizer :
      MulticlassModelBase,
      IDataStorageProvider,
      IModelStorageProvider,
      IModelTrainerProvider,
      ITextLoaderOptionsProvider
   {
      #region Fields
      /// <summary>
      /// Nome colonna path immagine
      /// </summary>
      private const string _imagePathColumnName = "ImagePath";
      /// <summary>
      /// Pipe di training
      /// </summary>
      [NonSerialized]
      private IEstimator<ITransformer> _pipe;
      /// <summary>
      /// Formato dati di training
      /// </summary>
      private TextLoader.Options _textLoaderOptions;
      /// <summary>
      /// Nome colonna data e ora immagine
      /// </summary>
      private const string _timestampColumnName = "Timestamp";
      #endregion
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Trainer del modello
      /// </summary>
      public IModelTrainer ModelTrainer { get; set; }
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
                  new TextLoader.Column(LabelColumnName, DataKind.String, 0),
                  new TextLoader.Column(_imagePathColumnName, DataKind.String, 1),
                  new TextLoader.Column(_timestampColumnName, DataKind.DateTime, 2),
               }
            };
         }
      }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public ImageRecognizer() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public ImageRecognizer(int? seed) : base(seed) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public ImageRecognizer(MachineLearningContext ml) : base(ml) { }
      /// <summary>
      /// Restituisce la pipe di training del modello
      /// </summary>
      /// <returns></returns>
      protected override IEstimator<ITransformer> GetPipe()
      {
         // Pipe di training
         return _pipe ??=
            ML.NET.Transforms.Conversion.MapValueToKey("Label", LabelColumnName).
            Append(ML.NET.Transforms.LoadRawImageBytes("Features", null, _imagePathColumnName)).
            Append(Trainers.ImageClassification()).
            Append(ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
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
      /// Ottiene un elenco di dati di training data dalla directory radice delle immagini.
      /// </summary>
      /// <param name="path">La directory radice delle immagini</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>La lista di dati di training</returns>
      /// <remarks>Le immagini vanno oganizzate in sottodirectory della radice, in cui il nome della sottodirectory specifica la label delle immagini contenute.</remarks>
      private async Task<DataViewGrid> GetTrainingDataFromPathAsync(string path, CancellationToken cancellation = default)
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
                       let line = new { Label = Path.GetFileName(dir), ImagePath = file, Timestamp = File.GetLastWriteTimeUtc(file) }
                       orderby line.ImagePath
                       select line;
            var dataGrid = DataViewGrid.Create(this, ML.NET.Data.CreateTextLoader(TextLoaderOptions).GetOutputSchema());
            foreach (var line in data) {
               dataGrid.Add((LabelColumnName, line.Label), (_imagePathColumnName, line.ImagePath), (_timestampColumnName, line.Timestamp));
               cancellation.ThrowIfCancellationRequested();
            }
            return dataGrid;
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
         var trainingData = await GetTrainingDataFromPathAsync(path, cancellation);
         cancellation.ThrowIfCancellationRequested();
         ML.NET.WriteLog("Updating image list", Name);
         // Effettua l'aggiornamento dello storage di dati sincronizzandolo con lo stato delle immagini
         await Task.Run(async () =>
         {
            // Immagini da scartare nello storage
            var invalidStorageImages = new HashSet<long>();
            // Immagini da scartare nel training
            var invalidTrainingImages = new HashSet<long>();
            // Task per parallelizzare i confronti
            var tasks = Enumerable.Range(0, Environment.ProcessorCount).Select(i => Task.CompletedTask).ToArray();
            var taskIx = 0;
            // Scandisce lo storage alla ricerca di elementi non piu' validi o aggiornati
            foreach (var cursor in DataStorage.LoadData(this).GetRowCursor(trainingData.Schema).AsEnumerable()) {
               cancellation.ThrowIfCancellationRequested();
               // Riga di dati di storage
               var storageRow = cursor.ToDataViewValuesRow(this);
               var position = cursor.Position;
               // Task di comparazione
               tasks[taskIx] = Task.Run(() =>
               {
                  // Verifica esistenza del file
                  if (!File.Exists(storageRow[_imagePathColumnName])) {
                     lock (invalidStorageImages)
                        invalidStorageImages.Add(position);
                  }
                  else {
                     // Verifica incrociata con i dati di training
                     foreach (var dataRow in trainingData) {
                        cancellation.ThrowIfCancellationRequested();
                        // Verifica se nel training esiste un immagine con lo stesso path del file nello storage
                        if (storageRow[_imagePathColumnName] == dataRow[_imagePathColumnName]) {
                           // Verifica se i dati relativi all'immagine sono variati
                           if (storageRow.ToString() != dataRow.ToString()) {
                              lock (invalidStorageImages)
                                 invalidStorageImages.Add(position);
                              break;
                           }
                           else {
                              lock (invalidTrainingImages)
                                 invalidTrainingImages.Add(dataRow.Position);
                           }
                        }
                     }
                  }
               }, cancellation);
               // Attende i task
               if (++taskIx == tasks.Length) {
                  await Task.WhenAll(tasks);
                  taskIx = 0;
               }
            }
            // Attende termine di tutti i task
            cancellation.ThrowIfCancellationRequested();
            await Task.WhenAll(tasks);
            // Verifica se deve aggiornare lo storage
            if (invalidStorageImages.Count > 0 || invalidTrainingImages.Count == 0) {
               // Crea la vista dati mergiata e filtrata
               var mergedDataView =
                  DataStorage.LoadData(this).ToDataViewFiltered(cursor => !invalidStorageImages.Contains(cursor.Position)).
                  Merge(trainingData.ToDataViewFiltered(cursor => !invalidTrainingImages.Contains(cursor.Position)));
               cancellation.ThrowIfCancellationRequested();
               // File temporaneo per il merge
               using var mergedStorage = new DataStorageBinaryTempFile();
               // Salva il mix di dati nel file temporaneo
               mergedStorage.SaveData(this, mergedDataView);
               // Salva il file temporaneo nello storage
               mergedDataView = mergedStorage.LoadData(this);
               DataStorage.SaveData(this, mergedStorage.LoadData(this));
            }
         }, cancellation);
         cancellation.ThrowIfCancellationRequested();
      }
      #endregion
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class ImageRecognizer // Prediction
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
         internal Prediction(IDataAccess data)
         {
            var grid = data.ToDataViewGrid();
            Kind = grid[0]["PredictedLabel"];
            var scores = (float[])grid[0]["Score"];
            var slotNames = grid.Schema["Score"].GetSlotNames();
            Scores = slotNames.Zip(scores).Select(item => new KeyValuePair<string, float>(item.First, item.Second)).ToArray();
            Score = Scores.FirstOrDefault(s => s.Key == Kind).Value;
         }
         #endregion
      }
   }
}
