using MachineLearning.Data;
using MachineLearning.Model;
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
      IInputSchemaProvider,
      IDataStorageProvider,
      IModelStorageProvider,
      IModelTrainerProvider,
      ITextLoaderOptionsProvider
   {
      #region Fields
      /// <summary>
      /// Pipe di training
      /// </summary>
      [NonSerialized]
      private ModelPipes _pipes;
      #endregion
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Nome colonna path immagine
      /// </summary>
      public string ImagePathColumnName { get; set; } = "ImagePath";
      /// <summary>
      /// Nome colonna data e ora immagine
      /// </summary>
      public string ImageTimestampColumnName { get; set; } = "ImageTimestamp";
      /// <summary>
      /// Schema di input dei dati
      /// </summary>
      public DataViewSchema InputSchema { get; set; }
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Trainer del modello
      /// </summary>
      public IModelTrainer ModelTrainer { get; set; } = new ModelTrainerStandard();
      /// <summary>
      /// Opzioni di caricamento dati in formato testo
      /// </summary>
      public TextLoader.Options TextLoaderOptions => new TextLoader.Options
      {
         AllowQuoting = true,
         Separators = new[] { ',' },
         Columns = InputSchema.ToTextLoaderColumns(),
      };
      #endregion
         #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public ImageRecognizer(MachineLearningContext ml = default) : base(ml) =>
         InputSchema = DataViewSchemaBuilder.Build(("Label", typeof(string)), (ImagePathColumnName, typeof(string)), (ImageTimestampColumnName, typeof(DateTime)));
      /// <summary>
      /// Restituisce le pipe di training del modello
      /// </summary>
      /// <returns>Le pipe</returns>
      public override ModelPipes GetPipes()
      {
         // Pipe di training
         return _pipes ??= new ModelPipes
         {
            Input =
               ML.NET.Transforms.Conversion.MapValueToKey("Label", LabelColumnName)
               .Append(ML.NET.Transforms.LoadRawImageBytes("Features", null, ImagePathColumnName)),
            Trainer =
               Trainers.ImageClassification(),
            Output =
               ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel")
         };
      }
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <returns>Il tipo di immagine</returns>
      public Prediction GetPrediction(string imagePath) => GetPredictionAsync(imagePath, default).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <param name="cancel">Eventuale token di cancellazione</param>
      /// <returns>Il task di previsione del tipo di immagine</returns>
      public async Task<Prediction> GetPredictionAsync(string imagePath, CancellationToken cancel = default)
      {
         var schema = InputSchema;
         return new Prediction(await GetPredictionDataAsync(schema.Select(c => c.Name == ImagePathColumnName ? imagePath : null).ToArray(), cancel));
      }
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
            var inputSchema = InputSchema;
            var dataGrid = DataViewGrid.Create(this, inputSchema);
            var timestampColumnName = inputSchema.FirstOrDefault(c => c.Name == ImageTimestampColumnName);
            foreach (var line in data) {
               var values = new List<(string Name, object Path)>
               {
                  (LabelColumnName, line.Label),
                  (ImagePathColumnName, line.ImagePath)
               };
               if (!string.IsNullOrWhiteSpace(ImageTimestampColumnName))
                  values.Add((ImageTimestampColumnName, line.Timestamp));
               dataGrid.Add(values.ToArray());
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
            foreach (var cursor in LoadData(DataStorage).GetRowCursor(trainingData.Schema).AsEnumerable()) {
               cancellation.ThrowIfCancellationRequested();
               // Riga di dati di storage
               var storageRow = cursor.ToDataViewValuesRow(this);
               var position = cursor.Position;
               // Task di comparazione
               tasks[taskIx] = Task.Run(() =>
               {
                  // Verifica esistenza del file
                  if (!File.Exists(storageRow[ImagePathColumnName])) {
                     lock (invalidStorageImages)
                        invalidStorageImages.Add(position);
                  }
                  else {
                     // Verifica incrociata con i dati di training
                     foreach (var dataRow in trainingData) {
                        cancellation.ThrowIfCancellationRequested();
                        // Verifica se nel training esiste un immagine con lo stesso path del file nello storage
                        if (storageRow[ImagePathColumnName] == dataRow[ImagePathColumnName]) {
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
                  LoadData(DataStorage).ToDataViewFiltered(cursor => !invalidStorageImages.Contains(cursor.Position)).
                  Merge(trainingData.ToDataViewFiltered(cursor => !invalidTrainingImages.Contains(cursor.Position)));
               cancellation.ThrowIfCancellationRequested();
               // File temporaneo per il merge
               using var mergedStorage = new DataStorageBinaryTempFile();
               // Salva il mix di dati nel file temporaneo
               SaveData(mergedStorage, mergedDataView);
               // Salva il file temporaneo nello storage
               mergedDataView = LoadData(mergedStorage);
               SaveData(DataStorage, LoadData(mergedStorage));
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
