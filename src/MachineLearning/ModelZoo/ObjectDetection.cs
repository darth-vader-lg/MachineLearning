using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Transforms;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using ODModelBuilderTF;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.ModelZoo
{
   /// <summary>
   /// Class for object detection in images
   /// </summary>
   [Serializable]
   public sealed partial class ObjectDetection :
      ModelZooBase<ObjectDetection.Mdl>,
      IDataStorageProvider,
      IInputSchema,
      IModelAutoSave,
      IModelStorageProvider // TODO: Implement the IModelTrainer
   {
      #region Properties
      /// <summary>
      /// Data storage
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// The batch size of the train (the one in the model configuration if null).
      /// </summary>
      public int? BatchSize { get; set; }
      /// <summary>
      /// Default resize of the nput images if not specified in the model
      /// </summary>
      public Size DefaultImageResize { get; set; } = new Size(640, 640);
      /// <summary>
      /// Path of the train folder. The temporary folder if null
      /// </summary>
      public string TrainFolder { get; set; }
      /// <summary>
      /// Input data schema
      /// </summary>
      public DataSchema InputSchema { get; private set; }
      /// <summary>
      /// The maximum accepted evaluation's total loss (if set).
      /// </summary>
      public double? MaxEvalTotalLoss { get; set; }
      /// <summary>
      /// The maximum accepted step's total loss (if set).
      /// </summary>
      public double? MaxStepTotalLoss { get; set; }
      /// <summary>
      /// Max number of train cycle. The one in the model's configuration file if not specified 
      /// </summary>
      public int? MaxTrainingCycles { get; set; }
      /// <summary>
      /// The minimum accepted average precision (if set).
      /// </summary>
      public double? MinEvalAveragePrecision { get; set; }
      /// <summary>
      /// Enable automatic save of updated model
      /// </summary>
      public bool ModelAutoSave => true;
      /// <summary>
      /// Model storage
      /// </summary>
      public IModelStorage ModelStorage { get; set; }
      /// <summary>
      /// Name od the model
      /// </summary>
      public string Name { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="context">Machine learning context provider</param>
      public ObjectDetection(IContextProvider<MLContext> context = default)
      {
         Model = new Mdl(this, context);
         SetSchema(0, 1, 2, "ImagePath", "ImageTimestamp", "Eval");
      }
      /// <summary>
      /// Clear the model
      /// </summary>
      public void ClearModel() => Model.ClearModel();
      /// <summary>
      /// Get the detected objects in an image
      /// </summary>
      /// <param name="imagePath">Path of the image</param>
      /// <param name="cancel">Optional cancellation token</param>
      /// <returns>The prediction containing the detected object boxes</returns>
      public Prediction GetPrediction(string imagePath, CancellationToken cancel = default)
      {
         var schema = InputSchema;
         return new(Model, Model.GetPredictionData(schema.Select(c => c.Name == Model.ImagePathColumnName ? imagePath : null).ToArray(), cancel));
      }
      /// <summary>
      /// Get a set of training data from a list of files
      /// </summary>
      /// <param name="trainImages">Set of train images</param>
      /// <param name="evalImages">Set of evaluation images</param>
      /// <param name="cancellation">Cancellation token</param>
      /// <returns>The train data set</returns>
      private DataViewGrid GetTrainingDataFromFiles(IEnumerable<string> trainImages, IEnumerable<string> evalImages, CancellationToken cancellation = default)
      {
         var inputSchema = InputSchema;
         var dataGrid = DataViewGrid.Create(Model, inputSchema);
         var timestampColumnName = inputSchema.FirstOrDefault(c => c.Name == Model.ImageTimestampColumnName);
         foreach (var (Paths, Eval) in new[] { (Paths: trainImages, Eval: false), (Paths: evalImages, Eval: true) }) {
            var data = from file in Paths
                       let ext = Path.GetExtension(file).ToLower()
                       where new[] { ".jpg", ".png", ".bmp", ".jfif" }.Contains(ext)
                       where File.Exists(Path.ChangeExtension(file, ".xml"))
                       let item = (Path: file, Timestamp: File.GetLastWriteTimeUtc(file), Eval)
                       orderby item.Path
                       select item;
            foreach (var item in data) {
               var values = new List<(string Name, object Value)>
                  {
                     (Model.ImagePathColumnName, item.Path),
                     (Model.ImageEvalColumnName, item.Eval)
                  };
               if (!string.IsNullOrWhiteSpace(Model.ImageTimestampColumnName))
                  values.Add((Model.ImageTimestampColumnName, item.Timestamp));
               dataGrid.Add(values.ToArray());
               cancellation.ThrowIfCancellationRequested();
            }
         }
         return dataGrid;
      }
      /// <summary>
      /// Set the data schema
      /// </summary>
      /// <param name="imagePathColumnIndex">Index of the column containing the path of the annotated image</param>
      /// <param name="imageTimestampColumnIndex">Index of the column containing the timestamp of the annotated image. Excluded colum if -1</param>
      /// <param name="imageEvalColumnIndex">Index of the column containing the evaluation specification of the image</param>
      /// <param name="columnsNames">Schema's column names</param>
      public void SetSchema(int imagePathColumnIndex = 0, int imageTimestampColumnIndex = 1, int imageEvalColumnIndex = 2, params string[] columnsNames)
      {
         if (imagePathColumnIndex < 0 || imagePathColumnIndex >= columnsNames.Length)
            throw new ArgumentException("The image path column index is out of range", nameof(imagePathColumnIndex));
         if (imageTimestampColumnIndex >= columnsNames.Length)
            throw new ArgumentException("The image timestamp column index is out of range", nameof(imageTimestampColumnIndex));
         if (imageEvalColumnIndex >= columnsNames.Length)
            throw new ArgumentException("The image for evaluation purpose column index is out of range", nameof(imageEvalColumnIndex));
         if (columnsNames.Any(item => string.IsNullOrEmpty(item)))
            throw new ArgumentException("All the columns must have a name", nameof(columnsNames));
         Model.ImagePathColumnName = columnsNames[imagePathColumnIndex];
         Model.ImageTimestampColumnName = imageTimestampColumnIndex > -1 ? columnsNames[imageTimestampColumnIndex] : null;
         Model.ImageEvalColumnName = columnsNames[imageEvalColumnIndex];
         var cTypes = Enumerable.Repeat(typeof(string), columnsNames.Length).ToArray();
         if (imageTimestampColumnIndex > -1)
            cTypes[imageTimestampColumnIndex] = typeof(DateTime);
         if (imageEvalColumnIndex > -1)
            cTypes[imageEvalColumnIndex] = typeof(bool);
         InputSchema = DataViewSchemaBuilder.Build(columnsNames.Select((c, i) => (c, cTypes[i])).ToArray());
      }
      /// <summary>
      /// Update the data storage with the specified images
      /// </summary>
      /// <param name="trainImages">Set of train images</param>
      /// <param name="evalImages">Set of evaluation images</param>
      /// <param name="cancellation">Cancellation token</param>
      public void UpdateStorageByImages(IEnumerable<string> trainImages, IEnumerable<string> evalImages, CancellationToken cancellation = default)
      {
         // Check if data storage is defined
         if (DataStorage == null)
            return;
         // Get train data
         var trainingData = GetTrainingDataFromFiles(trainImages, evalImages, cancellation);
         cancellation.ThrowIfCancellationRequested();
         // Update and syncronize the data storage by the state of the images
         // Images to discard in the storage
         var invalidStorageImages = new HashSet<long>();
         // Immagini to discard in the train
         var invalidTrainingImages = new HashSet<long>();
         // Task to parallelize the comparings
         var tasks = Enumerable.Range(0, Environment.ProcessorCount).Select(i => Task.CompletedTask).ToArray();
         var taskIx = 0;
         // Scan the storage to find no more valid or updated elements
         var currentData = Model.LoadData(DataStorage);
         foreach (var cursor in currentData?.GetRowCursor(trainingData.Schema).AsEnumerable() ?? Array.Empty<DataViewRowCursor>()) {
            cancellation.ThrowIfCancellationRequested();
            // Storage data row
            var storageRow = cursor.ToDataViewValuesRow(Model);
            var position = cursor.Position;
            // Compare tasks
            tasks[taskIx] = Task.Run(() =>
            {
               // Check for file existence
               if (!File.Exists(storageRow[Model.ImagePathColumnName])) {
                  lock (invalidStorageImages)
                     invalidStorageImages.Add(position);
               }
               else {
                  // Cross check with train data
                  foreach (var dataRow in trainingData) {
                     cancellation.ThrowIfCancellationRequested();
                     // Check if an image in the train set, with the same path, is contained in the storage
                     if (storageRow[Model.ImagePathColumnName] == dataRow[Model.ImagePathColumnName]) {
                        // Check if image's data changed
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
            // Wait tasks
            if (++taskIx == tasks.Length) {
               Task.WhenAll(tasks).Wait(cancellation);
               taskIx = 0;
            }
         }
         // wait for all tasks
         cancellation.ThrowIfCancellationRequested();
         Task.WhenAll(tasks).Wait(cancellation);
         // Check if the storage update is needed
         if (invalidStorageImages.Count > 0 || (invalidTrainingImages.Count == 0 && (trainingData.GetRowCount() ?? 0L) > 0)) {
            // Create the merged and filtered data view
            var filteredTrainingData = trainingData.ToDataViewFiltered(cursor => !invalidTrainingImages.Contains(cursor.Position));
            var mergedDataView = currentData != null ? currentData.ToDataViewFiltered(cursor => !invalidStorageImages.Contains(cursor.Position)).Merge(filteredTrainingData) : filteredTrainingData;
            cancellation.ThrowIfCancellationRequested();
            // Temporary file for the merge
            using var mergedStorage = new DataStorageBinaryTempFile();
            // Save the mixed data in the temporary storage
            Model.SaveData(mergedStorage, mergedDataView);
            // Save the temporary storage in the official storage
            mergedDataView = Model.LoadData(mergedStorage);
            Model.SaveData(DataStorage, Model.LoadData(mergedStorage));
         }
         cancellation.ThrowIfCancellationRequested();
      }
      /// <summary>
      /// Update the data storage with the specified image folders
      /// </summary>
      /// <param name="trainFolders">Set of train image folders</param>
      /// <param name="evalFolders">Set of evaluation image folders</param>
      /// <param name="cancellation">Cancellation token</param>
      public void UpdateStorageByFolders(IEnumerable<string> trainFolders, IEnumerable<string> evalFolders, CancellationToken cancellation = default)
      {
         var trainImages =
            from folder in trainFolders
            from file in Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)
            select file;
         var evalImages =
            from folder in evalFolders
            from file in Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)
            select file;
         UpdateStorageByImages(trainImages, evalImages, cancellation);
      }
      #endregion
   }

   /// <summary>
   /// Model
   /// </summary>
   public sealed partial class ObjectDetection // Mdl
   {
      [Serializable]
      public sealed class Mdl :
         ModelBaseMLNet,
         IDataStorageProvider,
         IDataTransformer,
         IInputSchema,
         IModelName,
         IModelAutoSave,
         IModelStorageProvider,
         IModelTrainerCycling,
         IModelTrainingStandard,
         ITextLoaderOptions
      {
         #region Fields
         /// <summary>
         /// Reference dictionary for detection data types -> output column number
         /// </summary>
         private readonly Dictionary<string, int> dataKindToColumn = new();
         /// <summary>
         /// Path of the last exported model
         /// </summary>
         [NonSerialized]
         private string _lastExportedModelPath = null;
         /// <summary>
         /// New model available event
         /// </summary>
         [NonSerialized]
         private AutoResetEvent _newModelAvailable;
         /// <summary>
         /// Owner object
         /// </summary>
         private readonly ObjectDetection _owner;
         /// <summary>
         /// Training pipe
         /// </summary>
         [NonSerialized]
         private ModelPipes _pipes;
         /// <summary>
         /// Train folder
         /// </summary>
         [NonSerialized]
         private string _trainFolder;
         /// <summary>
         /// Train task
         /// </summary>
         [NonSerialized]
         private readonly CancellableTask _trainTask = new();
         #endregion
         #region Properties
         /// <summary>
         /// Model configuration
         /// </summary>
         [field: NonSerialized]
         internal ODModelConfig Config { get; private set; }
         /// <summary>
         /// Data storage
         /// </summary>
         public IDataStorage DataStorage => ((IDataStorageProvider)_owner).DataStorage;
         /// <summary>
         /// Input schema
         /// </summary>
         DataSchema IInputSchema.InputSchema => ((IInputSchema)_owner).InputSchema;
         /// <summary>
         /// Name of the image's path column
         /// </summary>
         public string ImagePathColumnName { get; set; }
         /// <summary>
         /// Name of the image's purpose (train/eval) column
         /// </summary>
         public string ImageEvalColumnName { get; set; }
         /// <summary>
         /// Name of the image's timestamp column
         /// </summary>
         public string ImageTimestampColumnName { get; set; }
         /// <summary>
         /// Max number of train steps. The one in the model's configuration if not specified
         /// </summary>
         int IModelTrainerCycling.MaxTrainingCycles => _owner.MaxTrainingCycles ?? -1;
         /// <summary>
         /// Labels
         /// </summary>
         [field: NonSerialized]
         internal ReadOnlyCollection<string> Labels { get; set; }
         /// <summary>
         /// Enable automatic save of updated model
         /// </summary>
         public bool ModelAutoSave => ((IModelAutoSave)_owner).ModelAutoSave;
         /// <summary>
         /// Model's name
         /// </summary>
         public string ModelName => _owner.Name;
         /// <summary>
         /// Model storage
         /// </summary>
         public IModelStorage ModelStorage => ((IModelStorageProvider)_owner).ModelStorage;
         /// <summary>
         /// Opzioni di caricamento dati in formato testo
         /// </summary>
         public TextLoader.Options TextLoaderOptions => new()
         {
            AllowQuoting = true,
            Separators = new[] { ',' },
            Columns = ((IInputSchema)this).InputSchema.ToTextLoaderColumns(),
         };
         #endregion
         #region Methods
         /// <summary>
         /// Constructor
         /// </summary>
         /// <param name="owner">Owner object</param>
         /// <param name="contextProvider">Machine learning context provider</param>
         internal Mdl(ObjectDetection owner, IContextProvider<MLContext> contextProvider = default) : base(contextProvider) => _owner = owner;
         /// <summary>
         /// Dispose function
         /// </summary>
         /// <param name="disposing">Indicatore di dispose da codice</param>
         protected sealed override void Dispose(bool disposing)
         {
            if (IsDisposed)
               return;
            if (disposing) {
               try {
                  _pipes?.Dispose();
               }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
               }
            }
            _pipes = null;
            base.Dispose(disposing);
         }
         /// <summary>
         /// Return the column's index of the requested data kind
         /// </summary>
         /// <param name="dataKind">Data kind</param>
         /// <returns>The column index</returns>
         internal int GetColumnIndex(string dataKind)
         {
            lock (dataKindToColumn) {
               if (dataKindToColumn.Count == 0) {
                  // Schema di outputput del modello
                  var outputSchema = GetOutputSchema(_owner.InputSchema);
                  // Rileva con un dizionario intelligente gli indici delle colonne di uscita
                  var sd = new SmartDictionary<string>(from c in outputSchema select new KeyValuePair<string, string>(c.Name, c.Name));
                  // Aggiorna il dizionario di corrispondenze
                  new KeyValuePair<string, int>[] {
                     new(nameof(Prediction.DetectionBoxes), outputSchema[sd.Similar["detection boxes"]].Index),
                     new(nameof(Prediction.DetectionClasses), outputSchema[sd.Similar["detection classes"]].Index),
                     new(nameof(Prediction.DetectionScores), outputSchema[sd.Similar["detection scores"]].Index),
                  }.ToList().ForEach(item => dataKindToColumn.Add(item.Key, item.Value));
               }
               return dataKindToColumn.TryGetValue(dataKind, out var index) ? index : -1;
            }
         }
         /// <summary>
         /// Get the training pipes of the model
         /// </summary>
         /// <returns>The pipes</returns>
         public sealed override ModelPipes GetPipes() => _pipes ??= GetPipesInternal(Config);
         /// <summary>
         /// Get the training pipes of the model
         /// </summary>
         /// <param name="config">Model configuration</param>
         /// <param name="modelFilePath">Path of the model file. The one in config if null</param>
         /// <returns>The pipes</returns>
         private ModelPipes GetPipesInternal(ODModelConfig config, string modelFilePath = null)
         {
            // Path of the model
            modelFilePath ??= config.ModelFilePath;
            // Check if it's a Yolo model
            var isYolo = config.ModelType.ToLower().Contains("yolo");
            // Dimensions to use with the input tensor if not defined in the model
            var resize = new Size(
               config.ImageSize.Width > 0 ? config.ImageSize.Width : _owner.DefaultImageResize.Width > 0 ? _owner.DefaultImageResize.Width : 640,
               config.ImageSize.Height > 0 ? config.ImageSize.Height : _owner.DefaultImageResize.Height > 0 ? _owner.DefaultImageResize.Height : 640);
            var shapeDims = new List<int>();
            if (config.Inputs[0].Dim[0] < 0)
               shapeDims.Add(config.Inputs[0].Dim[0]);
            shapeDims.AddRange(!isYolo ? new[] { resize.Width, resize.Height } : new[] { resize.Height, resize.Width });
            var shapes = new Queue<int>(shapeDims);
            // Create the output pipeline
            var outputEstimators = new EstimatorList();
            var dropColumns = new HashSet<string>();
            // Yolo models output mapping
            if (isYolo) {
               // Transform Yolov5 output to a standard output
               var inputColumnName = config.Outputs.Where(o => o.Dim.Length == 3).FirstOrDefault()?.ColumnName ?? "detection";
               outputEstimators.Add(Context.Transforms.ScoreYolov5(inputColumnName: inputColumnName));
               // Remove Yolo model's output columns
               (from c in config.Outputs select c.ColumnName).ToList().ForEach(c => dropColumns.Add(c));
            }
            // Column to transform if the name of the tensor in the model doesn't match the relative ML.NET column
            var columnNameTransform = (
               from g in new[] { config.Inputs, config.Outputs }
               from c in g
               where c.Name != c.ColumnName
               select c).ToArray();
            // Reneame the columns
            foreach (var c in columnNameTransform)
               outputEstimators.Add(Context.Transforms.CopyColumns(inputColumnName: c.ColumnName, outputColumnName: c.Name));
            // Add the column to remove at the end of the set
            columnNameTransform.ToList().ForEach(c => dropColumns.Add(c.ColumnName));
            dropColumns.Add("Image");
            dropColumns.Add("ResizedImage");
            config.Inputs.ToList().ForEach(c => dropColumns.Add(c.ColumnName));
            // Drop the colums
            if (dropColumns.Count > 0)
               outputEstimators.Add(Context.Transforms.DropColumns(dropColumns.ToArray()));
            // Add the labels
            outputEstimators.Add(Context.Transforms.AddConst("Labels", $"\"{string.Join("\n", config.Labels)}\""));
            // Return the three pipelines
            return new()
            {
               // Data input pipe
               Input =
                  Context.Transforms.LoadImages(
                     inputColumnName: "ImagePath",
                     outputColumnName: "Image",
                     imageFolder: "")
                  .Append(Context.Transforms.ResizeImages(
                     inputColumnName: "Image",
                     outputColumnName: "ResizedImage",
                     imageWidth: resize.Width,
                     imageHeight: resize.Height,
                     resizing: ImageResizingEstimator.ResizingKind.Fill))
                  .Append(Context.Transforms.ExtractPixels(
                     inputColumnName: "ResizedImage",
                     outputColumnName: config.Inputs[0].ColumnName,
                     scaleImage: !isYolo ? 1f : 1f / 255f,
                     interleavePixelColors: !isYolo,
                     outputAsFloatArray: config.Inputs[0].DataType == typeof(float))),
               // Model inference pipe
               Trainer =
                  config.Format switch
                  {
                     // Onnx
                     ODModelConfig.ModelFormat.Onnx =>
                        Context.Transforms.ApplyOnnxModel(
                           inputColumnNames: new[] { config.Inputs[0].ColumnName },
                           outputColumnNames: (from c in config.Outputs select c.ColumnName).ToArray(),
                           modelFile: modelFilePath,
                           shapeDictionary: new Dictionary<string, int[]>()
                           {
                              {
                                 config.Inputs[0].ColumnName,
                                 config.Inputs[0].Dim.Select(d => d > 0 ? d : shapes.Dequeue()).ToArray()
                              }
                           },
                           recursionLimit: 100),
                     // saved_model o frozen graph TensorFlow
                     var tf when tf == ODModelConfig.ModelFormat.TF2SavedModel || tf == ODModelConfig.ModelFormat.TFFrozenGraph =>
                        Context.Model.LoadTensorFlowModel(modelFilePath).ScoreTensorFlowModel(
                           inputColumnNames: new[] { config.Inputs[0].ColumnName },
                           outputColumnNames: (from c in config.Outputs select c.ColumnName).ToArray()),
                     _ => throw new FormatException("Unknown model format")
                  },
               // Output pipe
               Output = outputEstimators.GetPipe()
            };
         }
         /// <summary>
         /// Get the model executing the train
         /// </summary>
         /// <param name="model">Model to train with</param>
         /// <param name="data">Train data</param>
         /// <param name="evaluationMetrics">Optional pre-computed evaluation metrics</param>
         /// <param name="cancellation">Cancellation token</param>
         /// <returns>The trained model</returns>
         IDataTransformer IModelTrainer.GetTrainedModel(ModelBase model, IDataAccess data, out object evaluationMetrics, CancellationToken cancellation)
         {
            // Trainer parameters
            _trainFolder ??= _owner.TrainFolder;
            if (_trainFolder == null) {
               _trainFolder = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
               Directory.CreateDirectory(_trainFolder);
            }
            var trainerOpt = new Trainer.Options
            {
               BatchSize = _owner.BatchSize,
               CheckpointEvery = int.MaxValue,
               //CheckpointForceThreashold = 0.95,
               EvalImagesFolder = Path.Combine(_trainFolder, "images", "eval"),
               ExportFolder = Path.Combine(_trainFolder, "export"),
               ModelType = ModelTypes.SSD_MobileNet_V2_320x320,
               NumTrainSteps = ((IModelTrainerCycling)this).MaxTrainingCycles,
               OnnxModelFileName = "Model.onnx",
               TensorBoardPort = 8080,
               TrainFolder = Path.Combine(_trainFolder, "train"),
               TrainImagesFolder = Path.Combine(_trainFolder, "images", "train"),
               TrainRecordsFolder = Path.Combine(_trainFolder, "records"),
            };
            evaluationMetrics = null;
            if (_trainTask.Task.IsCompleted || _trainTask.CancellationToken.IsCancellationRequested) {
               if (_trainTask.CancellationToken.IsCancellationRequested)
                  _trainTask.Task.Wait(cancellation);
               cancellation.ThrowIfCancellationRequested();
               _newModelAvailable?.Dispose();
               _newModelAvailable = new AutoResetEvent(false);
               _trainTask.StartNew(cancellation => Task.Run(() =>
               {
                  // Log function
                  void Log(LogEventArgs e)
                  {
                     try {
                        using var reader = new StringReader(e.Message);
                        for (var line = reader.ReadLine(); line != null; line = reader.ReadLine()) {
                           if (!string.IsNullOrEmpty(line))
                              Channel.Trace(Microsoft.ML.Runtime.MessageSensitivity.Unknown, line);
                        }
                     }
                     catch (Exception exc) {
                        Trace.WriteLine(exc);
                     }
                  }
                  // Train cycle
                  try {
                     ODModelBuilderTF.ODModelBuilderTF.Log += Log;
                     // Update the images
                     var data = ((IDataStorageProvider)this)?.DataStorage;
                     if (data != null) {
                        var updateMessage = false;
                        var images = DataViewGrid.Create(LoadData(((IDataStorageProvider)this).DataStorage));
                        foreach (var row in images.Rows) {
                           var file = (string)row[ImagePathColumnName];
                           var dir = Path.GetDirectoryName(file).ToLower();
                           var dest = (row[ImageEvalColumnName] ? trainerOpt.EvalImagesFolder : trainerOpt.TrainImagesFolder).ToLower();
                           if (dir != dest) {
                              if (!updateMessage) {
                                 Channel.WriteLog("Update the images");
                                 updateMessage = true;
                              }
                              if (!Directory.Exists(dest))
                                 Directory.CreateDirectory(dest);
                              Channel.WriteLog($"Copying {Path.GetFileName(file)} to {dest}");
                              File.Copy(file, Path.Combine(dest, Path.GetFileName(file)), true);
                              file = Path.ChangeExtension(file, ".xml");
                              File.Copy(file, Path.Combine(dest, Path.GetFileName(file)), true);
                           }
                        }
                     }
                     // State of the training
                     var state = new TrainState();
                     // Check if it's available a restart from the latest exported checkpoint
                     var latestCheckpointDir = Path.Combine(trainerOpt.ExportFolder, "checkpoint");
                     if (File.Exists(Path.Combine(latestCheckpointDir, "checkpoint"))) {
                        // Delete all the old train files
                        Channel.WriteLog($"Found an exported checkpoint in {latestCheckpointDir}. Restarting from it.");
                        trainerOpt.PreTrainedModelDir = trainerOpt.ExportFolder;
                        if (Directory.Exists(trainerOpt.TrainFolder)) {
                           var oldTrainFiles =
                              Directory.GetFiles(trainerOpt.TrainFolder, "ckpt-*.*")
                              .Concat(new[] { Path.Combine(trainerOpt.TrainFolder, "checkpoint") })
                              .Concat(Directory.GetFiles(trainerOpt.TrainFolder).Where(f => Path.GetFileName(f).Contains(".export.")));
                           foreach (var file in oldTrainFiles) {
                              try { File.Delete(file); } catch { }
                           }
                        }
                        var stateFilePath =
                           string.IsNullOrEmpty(trainerOpt.OnnxModelFileName) ?
                           Path.Combine(trainerOpt.ExportFolder, "saved_model.metrics") :
                           Path.Combine(trainerOpt.ExportFolder, trainerOpt.OnnxModelFileName + ".metrics");
                        try {
                           state = JsonSerializer.Deserialize<TrainState>(File.ReadAllText(stateFilePath));
                           Channel.WriteLog($"Restored train state:");
                           foreach (var line in File.ReadAllLines(stateFilePath))
                              Channel.WriteLog(line);
                        }
                        catch (Exception) {
                           state = new TrainState();
                        }
                        // Read the current state
                        // Copy the last exported checkpoint in the train folder
                        if (!Directory.Exists(trainerOpt.TrainFolder))
                           Directory.CreateDirectory(trainerOpt.TrainFolder);
                        foreach (var file in Directory.GetFiles(latestCheckpointDir)) {
                           try { File.Copy(file, Path.Combine(trainerOpt.TrainFolder, Path.GetFileName(file)), true); } catch { }
                        }
                        trainerOpt.PreTrainedModelDir = null;
                     }
                     // Create the trainer
                     var trainer = new Trainer(trainerOpt);
                     // Best status
                     var bestStep = default(int?);
                     var bestStepLoss = state.StepLoss;
                     var latestChkStep = default(int?);
                     // Train step event
                     trainer.TrainStep += (sender, e) =>
                     {
                        // Training cancellation
                        if (e.Cancel = cancellation.IsCancellationRequested)
                           return;
                        // Update the status and write the message
                        state.StepNumber = e.StepNumber;
                        state.StepLoss = e.TotalLoss;
                        bestStepLoss ??= e.TotalLoss;
                        var sb = new StringBuilder();
                        sb.Append($"Step number:{e.StepNumber}");
                        sb.Append($" | Total loss:{e.TotalLoss:N3}");
                        sb.Append($" | Best loss:{bestStepLoss:N3}");
                        sb.Append($" | Ratio:{e.TotalLoss * 100.0 / bestStepLoss:###.0}%");
                        sb.Append($" | Step time:{e.StepTime:N3} secs");
                        if (bestStep != null)
                           sb.Append($" | Best step:{bestStep}");
                        Channel.WriteLog(sb.ToString());
                        // Force checkpoint every n steps just for tracing
                        if (latestChkStep != null && (e.StepNumber - latestChkStep >= 1000)) {
                           Channel.WriteLog($"Trace checkpoint at step {e.StepNumber}");
                           e.CreateCheckpoint = true;
                        }
                        // Check the minimum total loss required
                        if (_owner.MaxStepTotalLoss != null && e.TotalLoss > _owner.MaxStepTotalLoss)
                           return;
                        // Check the ratio between current loss and best loss
                        var ratio = e.TotalLoss / bestStepLoss;
                        if (ratio > 1.1)
                           return;
                        // Generate the checkpoint
                        if (!e.CreateCheckpoint) {
                           e.CreateCheckpoint = true;
                           Channel.WriteLog($"Create checkpoint with total loss {e.TotalLoss:N3}");
                        }
                        if (ratio < 1.0)
                           bestStepLoss = e.TotalLoss;
                     };
                     // Checkpoint event
                     trainer.Checkpoint += (sender, e) =>
                     {
                        if (state.StepNumber != null && state.StepLoss != null)
                           Channel.WriteLog($"Checkpoint created with total loss {state.StepLoss:N3} at {e.LatestCheckpointPath} ");
                        else
                           Channel.WriteLog($"Checkpoint created at {e.LatestCheckpointPath} ");
                        latestChkStep = state.StepNumber;
                     };
                     // Evaluation complete event
                     trainer.TrainEvaluation += (sender, e) =>
                     {
                        // Training cancellation
                        if (e.Cancel = cancellation.IsCancellationRequested)
                           return;
                        // First evaluation
                        if (state.EvalPrecision == null) {
                           state.EvalPrecision = e.AP;
                           Channel.WriteLog($"Current evaluation average precision: {state.EvalPrecision:N3}");
                        }
                        // Check if the model has to be exported due to better average precision
                        else {
                           var export = true;
                           if (_owner.MaxEvalTotalLoss != null && e.TotalLoss > _owner.MaxEvalTotalLoss) {
                              export = false;
                              Channel.WriteLog($"Discard model with evaluation total loss {e.TotalLoss:N3}. Maximum accepted is {_owner.MaxEvalTotalLoss:N3}");
                           }
                           if (export && _owner.MinEvalAveragePrecision != null && e.AP < _owner.MinEvalAveragePrecision) {
                              Channel.WriteLog($"Discard model with average precision {e.AP:N3}. Minimum average precision is {_owner.MinEvalAveragePrecision:N3}");
                              export = false;
                           }
                           if (export && e.AP < state.EvalPrecision + 0.0005) {
                              Channel.WriteLog($"Discard model with average precision {e.AP:N3}. Best average precision was {state.EvalPrecision:N3}");
                              export = false;
                           }
                           if (!export)
                              return;
                           e.Export = true;
                           bestStep = state.StepNumber;
                           bestStepLoss = state.StepLoss;
                           state.EvalLoss = e.TotalLoss;
                           state.EvalPrecision = e.AP;
                           state.CocoMetrics = new(e.Metrics);
                           Channel.WriteLog("Export model with metrics:");
                           foreach (var m in e.Metrics)
                              Channel.WriteLog($"{m.Key.PadRight(40, '.')}{m.Value:N3}");
                        }
                     };
                     // Export event management function
                     void OnExport(ExportEventArgs e)
                     {
                        // Test for train cancellation
                        if (e.Cancel = cancellation.IsCancellationRequested)
                           return;
                        // Write the log
                        Channel.WriteLog($"{e.Path} model available having average precision {state.EvalPrecision:N3}");
                        // Save the metrics
                        File.WriteAllText(e.Path + ".metrics", JsonSerializer.Serialize(state, new JsonSerializerOptions(JsonSerializerDefaults.General) { WriteIndented = true }));
                     }
                     // Saved model exported event
                     trainer.ExportedSavedModel += (sender, e) => OnExport(e);
                     // Onnx model exported event
                     trainer.ExportedOnnx += (sender, e) => OnExport(e);
                     // Set the available model at the endo of configuration export
                     if (string.IsNullOrEmpty(trainer.Opt.OnnxModelFileName))
                        trainer.ExportedSavedModelConfig += (sender, e) => _newModelAvailable.Set();
                     else
                        trainer.ExportedOnnxConfig += (sender, e) => _newModelAvailable.Set();
                     // Train
                     trainer.Train(cancellation);
                  }
                  catch (Exception) {
                     throw;
                  }
                  finally {
                     _trainTask.Cancel();
                     ODModelBuilderTF.ODModelBuilderTF.Log -= Log;
                  }
               }, cancellation), cancellation);
            }
            // Wait a new model or a train cancellation
            var ev = WaitHandle.WaitAny(new[] { _newModelAvailable, _trainTask.CancellationToken.WaitHandle });
            // Wait the task to propagate possible exceptions
            if (ev != 0)
               _trainTask.Task.Wait(CancellationToken.None);
            // Re-import and return the new model
            cancellation.ThrowIfCancellationRequested();
            if (ev != 0)
               return null;
            // Load the model configuration
            var config = ODModelConfig.Load(Path.Combine(trainerOpt.ExportFolder, trainerOpt.OnnxModelFileName));
            // Check the model
            if (config.Format == ODModelConfig.ModelFormat.Unknown)
               return null;
            // Create a copy of the model file
            var exportedModelPath = Path.Combine(trainerOpt.TrainFolder, Guid.NewGuid().ToString() + ".export.onnx");
            File.Copy(Path.Combine(trainerOpt.ExportFolder, trainerOpt.OnnxModelFileName), exportedModelPath);
            // Get the pipes
            var pipelines = GetPipesInternal(config, exportedModelPath);
            // Create the model
            var dataView = DataViewGrid.Create(this, _owner.InputSchema);
            var transformer = pipelines.Merged.Fit(dataView);
            var result = new DataTransformer<MLContext>(this, transformer);
            Config = config;
            _pipes?.Dispose();
            _pipes = pipelines;
            _lastExportedModelPath = exportedModelPath;
            return result;
         }
         /// <summary>
         /// Import an external model
         /// </summary>
         /// <param name="modelStorage">Storage of the model</param>
         /// <param name="schema">The model's schema</param>
         /// <returns>null</returns>
         public sealed override IDataTransformer ImportModel(IModelStorage modelStorage, out DataSchema schema)
         {
            // Initialize the schema to null
            schema = null;
            // Check if the importing model exists
            if (string.IsNullOrEmpty(modelStorage.ImportPath) || !File.Exists(modelStorage.ImportPath))
               return null;
            // Load the model configuration
            Config = ODModelConfig.Load(modelStorage.ImportPath);
            // Check if it's a known model type
            if (Config.Format == ODModelConfig.ModelFormat.Unknown)
               return null;
            // Check if the ML.NET format model is more recent than the model to import
            if (modelStorage is IDataTimestamp modelTimestamp && modelTimestamp.DataTimestamp >= File.GetLastWriteTimeUtc(Config.ModelFilePath))
               return null;
            // Get the pipes
            _pipes?.Dispose();
            _pipes = null;
            var pipelines = GetPipes();
            // Create the model
            var dataView = DataViewGrid.Create(this, _owner.InputSchema);
            var model = pipelines.Merged.Fit(dataView);
            schema = _owner.InputSchema;
            var result = new DataTransformer<MLContext>(this, model);
            // Save the model.
            SaveModel(modelStorage, result, schema);
            result.Dispose();
            return null;
         }
         /// <summary>
         /// Model changed function
         /// </summary>
         /// <param name="e">Event arguments</param>
         protected sealed override void OnModelChanged(ModelTrainingEventArgs e)
         {
            base.OnModelChanged(e);
            try {
               // Azzera le pipes all'invalidazione del modello per essere ricostruite
               if (e.Evaluator.Model == null) {
                  _pipes?.Dispose();
                  _pipes = null;
                  lock (dataKindToColumn)
                     dataKindToColumn.Clear();
               }
               try {
                  if (!string.IsNullOrEmpty(_lastExportedModelPath)) {
                     // Folder of exported models by the train
                     var dir = Path.GetDirectoryName(_lastExportedModelPath);
                     // Delete old exported files
                     var files = Directory.GetFiles(dir)
                        .Where(f => Path.GetExtension(Path.ChangeExtension(f, null)).ToLower() == ".export" && f.ToLower() != _lastExportedModelPath.ToLower());
                     foreach (var f in files) {
                        try { File.Delete(f); } catch { }
                     }
                  }
               }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
               }
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
         }
         #endregion
      }
   }

   /// <summary>
   /// Prediction result
   /// </summary>
   public sealed partial class ObjectDetection // TrainState
   {
      [Serializable]
      private class TrainState
      {
         #region Properties
         /// <summary>
         /// All Coco metrics
         /// </summary>
         public Dictionary<string, double> CocoMetrics { get; set; }
         /// <summary>
         /// Evaluation loss
         /// </summary>
         public double? EvalLoss { get; set; }
         /// <summary>
         /// Average evaluation precision
         /// </summary>
         public double? EvalPrecision { get; set; }
         /// <summary>
         /// Step loss
         /// </summary>
         public double? StepLoss { get; set; }
         /// <summary>
         /// Step number
         /// </summary>
         public int? StepNumber { get; set; }
         #endregion
         #region Methods
         /// <summary>
         /// Constructor
         /// </summary>
         public TrainState()
         {
         }
         /// <summary>
         /// Load the metrics in json format
         /// </summary>
         /// <param name="path">Metrics file path</param>
         public static TrainState Load(string path)
         {
            var result = JsonSerializer.Deserialize<TrainState>(File.ReadAllText(path));
            return result;
         }
         /// <summary>
         /// Save the metrics in json format
         /// </summary>
         /// <param name="path">Metrics file path</param>
         public void Save(string path)
         {
            var json = JsonSerializer.Serialize(this);
            File.WriteAllText(path, json);
         }
         #endregion
      }
   }

   /// <summary>
   /// Prediction result
   /// </summary>
   public sealed partial class ObjectDetection // Prediction
   {
      [Serializable]
      public partial class Prediction
      {
         #region Properties
         /// <summary>
         /// Output per i riquadri di contenimento
         /// </summary>
         public float[] DetectionBoxes { get; }
         /// <summary>
         /// Output per le classi di previsione
         /// </summary>
         public float[] DetectionClasses { get; }
         /// <summary>
         /// Output per gli scores
         /// </summary>
         public float[] DetectionScores { get; }
         /// <summary>
         /// Path dell'immagine
         /// </summary>
         public string ImagePath { get; }
         /// <summary>
         /// Labels
         /// </summary>
         public ReadOnlyCollection<string> Labels { get; }
         #endregion
         #region Methods
         /// <summary>
         /// Constructor
         /// </summary>
         /// <param name="data">Prediction data</param>
         internal Prediction(Mdl owner, IDataAccess data)
         {
            // Cgeate the result data grid
            var grid = data.ToDataViewGrid();
            // Store results of prediction
            ImagePath = grid[0]["ImagePath"];
            DetectionBoxes = grid[0][owner.GetColumnIndex(nameof(DetectionBoxes))];
            DetectionClasses = grid[0][owner.GetColumnIndex(nameof(DetectionClasses))];
            DetectionScores = grid[0][owner.GetColumnIndex(nameof(DetectionScores))];
            // Store the labels
            if (owner.Labels == null)
               owner.Labels = ((string)grid[0]["Labels"]).Split('\n').ToList().AsReadOnly();
            Labels = owner.Labels;
         }
         /// <summary>
         /// Return the filtered bounding boxes
         /// </summary>
         /// <param name="minScore">Min score (0 ... 1)</param>
         /// <returns>The list of bounding box</returns>
         public List<Box> GetBoxes(double minScore = 0.0)
         {
            var result = new List<Box>();
            int startId = 1;
            for (var i = 0; i < DetectionScores.Length; i++) {
               if (DetectionScores[i] < minScore)
                  continue;
               if (DetectionClasses[i] < startId)
                  continue;
               var detectionClass = (int)DetectionClasses[i] - startId;
               result.Add(new Box(
                  detectionClass,
                  detectionClass > -1 && detectionClass < Labels.Count ? Labels[detectionClass] : detectionClass.ToString(),
                  DetectionScores[i],
                  DetectionBoxes[i * 4 + 1],
                  DetectionBoxes[i * 4 + 0],
                  DetectionBoxes[i * 4 + 3] - DetectionBoxes[i * 4 + 1],
                  DetectionBoxes[i * 4 + 2] - DetectionBoxes[i * 4 + 0]));
            }
            return result;
         }
         #endregion
      }

      public partial class Prediction // Box
      {
         [Serializable]
         public class Box
         {
            #region Properties
            /// <summary>
            /// Height of the bounding box
            /// </summary>
            public float Height { get; }
            /// <summary>
            /// Identifier of the object
            /// </summary>
            public int Id { get; }
            /// <summary>
            /// Left edge of the bounding box
            /// </summary>
            public float Left { get; }
            /// <summary>
            /// Name of the object
            /// </summary>
            public string Name { get; }
            /// <summary>
            /// Score
            /// </summary>
            public float Score { get; }
            /// <summary>
            /// Top edge of the bounding box
            /// </summary>
            public float Top { get; }
            /// <summary>
            /// Width of the bounding box
            /// </summary>
            public float Width { get; }
            #endregion
            #region Methods
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="id">Identifier of the object</param>
            /// <param name="name">Name of the object</param>
            /// <param name="score">Score</param>
            /// <param name="left">Left edge of the bounding box</param>
            /// <param name="top">Top edge of the bounding box</param>
            /// <param name="width">Width of the bounding box</param>
            /// <param name="height">Height of the bounding box</param>
            public Box(int id, string name, float score, float left, float top, float width, float height)
            {
               Id = id;
               Name = name;
               Height = height;
               Left = left;
               Score = score;
               Top = top;
               Width = width;
            }
            #endregion
         }
      }
   }
}
