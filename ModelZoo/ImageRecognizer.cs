using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.ModelZoo
{
   /// <summary>
   /// Class for image recognition
   /// </summary>
   [Serializable]
   public sealed partial class ImageRecognizer :
      ModelZooBase<ImageRecognizer.Mdl>,
      IDataStorageProvider,
      IInputSchema,
      IModelStorageProvider,
      IModelTrainerProvider
   {
      #region Properties
      /// <summary>
      /// Storage dei dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Directories sorgenti di immagini
      /// </summary>
      public string[] ImagesSources { get; set; }
      /// <summary>
      /// Schema di input dei dati
      /// </summary>
      public DataViewSchema InputSchema { get; private set; }
      /// <summary>
      /// Storage del modello
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Trainer del modello
      /// </summary>
      public IModelTrainer ModelTrainer { get; set; } = new ModelTrainerStandard();
      /// <summary>
      /// Nome del modello
      /// </summary>
      public string Name { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      public ImageRecognizer(IContextProvider<MLContext> context = default)
      {
         Model = new Mdl(this, context);
         SetSchema(0, 1, 2, "Label", "ImagePath", "ImageTimestamp");
      }
      /// <summary>
      /// Cancella il modello
      /// </summary>
      public void ClearModel() => Model.ClearModel();
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <param name="cancel">Eventuale token di cancellazione</param>
      /// <returns>Il task di previsione del tipo di immagine</returns>
      public Prediction GetPrediction(string imagePath, CancellationToken cancel = default)
      {
         var schema = InputSchema;
         return new Prediction(Model.GetPredictionData(schema.Select(c => c.Name == Model.ImagePathColumnName ? imagePath : null).ToArray(), cancel));
      }
      /// <summary>
      /// Imposta lo schema dei dati
      /// </summary>
      /// <param name="predictionColumnIndex">Indice della colonna di previsione</param>
      /// <param name="imagePathColumnIndex">Indice della colonna del path dell'immagine</param>
      /// <param name="imageTimestampColumnIndex">Indice della colonna del timestamp dell'immagine. Se -1 il timestamp verra' escluso</param>
      /// <param name="columnsNames">Nomi delle colonne dello schema</param>
      public void SetSchema(int predictionColumnIndex = 0, int imagePathColumnIndex = 1, int imageTimestampColumnIndex = 2, params string[] columnsNames)
      {
         if (predictionColumnIndex < 0 || predictionColumnIndex >= columnsNames.Length)
            throw new ArgumentException("The prediction column index is out of range", nameof(predictionColumnIndex));
         if (imagePathColumnIndex < 0 || imagePathColumnIndex >= columnsNames.Length)
            throw new ArgumentException("The image path column index is out of range", nameof(imagePathColumnIndex));
         if (imageTimestampColumnIndex >= columnsNames.Length)
            throw new ArgumentException("The image timestamp column index is out of range", nameof(imageTimestampColumnIndex));
         if (columnsNames.Any(item => string.IsNullOrEmpty(item)))
            throw new ArgumentException("All the columns must have a name", nameof(columnsNames));
         Model.LabelColumnName = columnsNames[predictionColumnIndex];
         Model.ImagePathColumnName = columnsNames[imagePathColumnIndex];
         Model.ImageTimestampColumnName = imageTimestampColumnIndex > -1 ? columnsNames[imageTimestampColumnIndex] : null;
         InputSchema = DataViewSchemaBuilder.Build(columnsNames.Select((c, i) => (c, i == imageTimestampColumnIndex ? typeof(DateTime) : typeof(string))).ToArray());
      }
      #endregion
   }

   /// <summary>
   /// Modello
   /// </summary>
   public sealed partial class ImageRecognizer // Mdl
   {
      [Serializable]
      public sealed class Mdl :
         MulticlassModelBase,
         IDataStorageProvider,
         IInputSchema,
         IModelAutoCommit,
         IModelAutoSave,
         IModelName,
         IModelStorageProvider,
         IModelTrainerProvider,
         ITextLoaderOptions
      {
         #region Fields
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly ImageRecognizer _owner;
         /// <summary>
         /// Pipe di training
         /// </summary>
         [NonSerialized]
         private ModelPipes _pipes;
         #endregion
         #region Properties
         /// <summary>
         /// Storage di dati
         /// </summary>
         public IDataStorage DataStorage => ((IDataStorageProvider)_owner).DataStorage;
         /// <summary>
         /// Nome colonna path immagine
         /// </summary>
         public string ImagePathColumnName { get; set; }
         /// <summary>
         /// Directory sorgenti di immagini
         /// </summary>
         public string[] ImagesSources { get; set; }
         /// <summary>
         /// Nome colonna data e ora immagine
         /// </summary>
         public string ImageTimestampColumnName { get; set; }
         /// <summary>
         /// Schema di input del modello
         /// </summary>
         public DataViewSchema InputSchema => ((IInputSchema)_owner).InputSchema;
         /// <summary>
         /// Abilitazione salvataggio automatico modello
         /// </summary>
         public bool ModelAutoCommit => true;
         /// <summary>
         /// Abilitazione commit automatico dei dati di training
         /// </summary>
         public bool ModelAutoSave => true;
         /// <summary>
         /// Nome del modello
         /// </summary>
         public string ModelName => _owner.Name;
         /// <summary>
         /// Storage del modello
         /// </summary>
         public IModelStorage ModelStorage => ((IModelStorageProvider)_owner).ModelStorage;
         /// <summary>
         /// Trainer del modello
         /// </summary>
         public IModelTrainer ModelTrainer => ((IModelTrainerProvider)_owner).ModelTrainer;
         /// <summary>
         /// Opzioni di caricamento dati in formato testo
         /// </summary>
         public TextLoader.Options TextLoaderOptions => new()
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
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="context">Contesto di machine learning</param>
         internal Mdl(ImageRecognizer owner, IContextProvider<MLContext> context) : base(context) => _owner = owner;
         /// <summary>
         /// Funzione di dispose
         /// </summary>
         /// <param name="disposing">Indicatore di dispose da codice</param>
         protected override void Dispose(bool disposing)
         {
            base.Dispose(disposing);
            try {
               _pipes?.Dispose();
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
            _pipes = null;
         }
         /// <summary>
         /// Restituisce le pipe di training del modello
         /// </summary>
         /// <returns>Le pipe</returns>
         public sealed override ModelPipes GetPipes()
         {
            // Pipe di training
            return _pipes ??= new ModelPipes
            {
               Input =
                  Context.Transforms.Conversion.MapValueToKey("Label", LabelColumnName)
                  .Append(Context.Transforms.LoadRawImageBytes("Features", null, ImagePathColumnName)),
               Trainer =
                  Trainers.ImageClassification(),
               Output =
                  Context.Transforms.Conversion.MapKeyToValue("PredictedLabel")
            };
         }
         /// <summary>
         /// Ottiene un elenco di dati di training data dalla directory radice delle immagini.
         /// </summary>
         /// <param name="dirs">Le directory radice delle immagini</param>
         /// <param name="cancellation">Token di cancellazione</param>
         /// <returns>La lista di dati di training</returns>
         /// <remarks>Le immagini vanno oganizzate in sottodirectory della radice, in cui il nome della sottodirectory specifica la label delle immagini contenute.</remarks>
         private DataViewGrid GetTrainingDataFromPath(string[] dirs, CancellationToken cancellation = default)
         {
            var folders = from dir in dirs ?? Array.Empty<string>()
                           from item in Directory.GetDirectories(dir, "*.*", SearchOption.TopDirectoryOnly)
                           where File.GetAttributes(item).HasFlag(FileAttributes.Directory)
                           select item;
            var imageData = from folder in folders
                              from file in Directory.GetFiles(folder, "*.*", SearchOption.TopDirectoryOnly)
                              let ext = Path.GetExtension(file).ToLower()
                              where new[] { ".jpg", ".png", ".bmp" }.Contains(ext)
                              let line = new { Label = Path.GetFileName(folder), ImagePath = file, Timestamp = File.GetLastWriteTimeUtc(file) }
                              orderby line.ImagePath
                              select line;
            var inputSchema = InputSchema;
            var dataGrid = DataViewGrid.Create(this, inputSchema);
            var timestampColumnName = inputSchema.FirstOrDefault(c => c.Name == ImageTimestampColumnName);
            foreach (var line in imageData) {
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
         }
         /// <summary>
         /// Funzione di start del training
         /// </summary>
         /// <param name="e"></param>
         protected sealed override void OnTrainingCycleStarted(ModelTrainingEventArgs e)
         {
            base.OnTrainingCycleStarted(e);
            if (e.Evaluator.Cancellation.IsCancellationRequested)
               return;
            UpdateStorage(_owner.ImagesSources, e.Evaluator.Cancellation);
         }
         /// <summary>
         /// Aggiorna lo storage di dati con l'elenco delle immagini categorizzate contenute nella directory indicata
         /// </summary>
         /// <param name="dirs">Le directory radice delle immagini</param>
         /// <param name="cancellation">Token di cancellazione</param>
         /// <remarks>Le immagini vanno oganizzate in sottodirectory della radice, in cui il nome della sottodirectory specifica la label delle immagini contenute.</remarks>
         private void UpdateStorage(string[] dirs, CancellationToken cancellation = default)
         {
            // Verifica che sia definito uno storage di dati
            if (DataStorage == null)
               return;
            // Ottiene i dati di training (l'elenco delle immagini classificate e datate)
            var trainingData = GetTrainingDataFromPath(dirs, cancellation);
            cancellation.ThrowIfCancellationRequested();
            Channel.WriteLog("Updating image list");
            // Effettua l'aggiornamento dello storage di dati sincronizzandolo con lo stato delle immagini
            // Immagini da scartare nello storage
            var invalidStorageImages = new HashSet<long>();
            // Immagini da scartare nel training
            var invalidTrainingImages = new HashSet<long>();
            // Task per parallelizzare i confronti
            var tasks = Enumerable.Range(0, Environment.ProcessorCount).Select(i => Task.CompletedTask).ToArray();
            var taskIx = 0;
            // Scandisce lo storage alla ricerca di elementi non piu' validi o aggiornati
            var currentData = LoadData(DataStorage);
            foreach (var cursor in currentData?.GetRowCursor(trainingData.Schema).AsEnumerable() ?? Array.Empty<DataViewRowCursor>()) {
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
                  Task.WhenAll(tasks).Wait(cancellation);
                  taskIx = 0;
               }
            }
            // Attende termine di tutti i task
            cancellation.ThrowIfCancellationRequested();
            Task.WhenAll(tasks).Wait(cancellation);
            // Verifica se deve aggiornare lo storage
            if (invalidStorageImages.Count > 0 || (invalidTrainingImages.Count == 0 && (trainingData.GetRowCount() ?? 0L) > 0)) {
               // Crea la vista dati mergiata e filtrata
               var filteredTrainingData = trainingData.ToDataViewFiltered(cursor => !invalidTrainingImages.Contains(cursor.Position));
               var mergedDataView = currentData != null ? currentData.ToDataViewFiltered(cursor => !invalidStorageImages.Contains(cursor.Position)).Merge(filteredTrainingData) : filteredTrainingData;
               cancellation.ThrowIfCancellationRequested();
               // File temporaneo per il merge
               using var mergedStorage = new DataStorageBinaryTempFile();
               // Salva il mix di dati nel file temporaneo
               SaveData(mergedStorage, mergedDataView);
               // Salva il file temporaneo nello storage
               mergedDataView = LoadData(mergedStorage);
               SaveData(DataStorage, LoadData(mergedStorage));
               Channel.WriteLog("Image list updated");
            }
            else
               Channel.WriteLog("Image list is still valid");
            cancellation.ThrowIfCancellationRequested();
         }
         #endregion
      }
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
