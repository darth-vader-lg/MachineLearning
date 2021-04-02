using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.TensorFlow;
using MachineLearning.Util;
using Microsoft.ML;
using NumSharp;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow;

namespace MachineLearning
{
   /// <summary>
   /// Classe per il rilevamento di oggetti nelle immagini
   /// </summary>
   [Serializable]
   public sealed partial class ObjectDetection :
      IInputSchema,
      IModelStorageProvider
   {
      #region Fields
      /// <summary>
      /// Modello
      /// </summary>
      private readonly Model _model;
      #endregion
      #region Properties
      /// <summary>
      /// Schema di input dei dati
      /// </summary>
      public DataViewSchema InputSchema { get; private set; }
      /// <summary>
      /// Storage del modello
      /// </summary>
      //public string ModelStorage { get; set; }
      public IModelStorage ModelStorage { get; set; }
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
      public ObjectDetection(IContextProvider<TFContext> context = default)
      {
         _model = new Model(this, context);
         InputSchema = DataViewSchemaBuilder.Build((Name: "ImagePath", Type: typeof(string)));
      }
      /// <summary>
      /// Cancella il modello
      /// </summary>
      public void ClearModel() => _model.ClearModel();
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <returns>Il tipo di immagine</returns>
      public Prediction GetPrediction(string imagePath) => GetPredictionAsync(imagePath, default).WaitSync();
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <param name="cancel">Eventuale token di cancellazione</param>
      /// <returns>Il task di previsione del tipo di immagine</returns>
      public async Task<Prediction> GetPredictionAsync(string imagePath, CancellationToken cancel = default) =>
         new Prediction(await _model.GetPredictionDataAsync(new[] { imagePath }, cancel));
      #endregion
   }

   /// <summary>
   /// Modello
   /// </summary>
   public sealed partial class ObjectDetection // Prediction
   {
      [Serializable]
      public sealed class Model :
         ModelBaseTensorFlow,
         IDataStorageProvider,
         IDataTransformer,
         IInputSchema,
         IModelName,
         IModelStorageProvider,
         IModelTrainerProvider
      {
         #region Fields
         /// <summary>
         /// Label degli oggetti conosciuti dal modello
         /// </summary>
         private PbtxtItems _labels;
         /// <summary>
         /// Soglia punteggio minimo
         /// </summary>
         private const float _minimumScore = 0.5f;
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly ObjectDetection _owner;
         /// <summary>
         /// Sessione TensorFlow
         /// </summary>
         private Session _session;
         /// <summary>
         /// Processo di training
         /// </summary>
         private Process _trainProcess;
         #endregion
         #region Properties
         /// <summary>
         /// Storage di dati
         /// </summary>
         public IDataStorage DataStorage { get; } = new DataStorageBinaryMemory();
         /// <summary>
         /// Schema di input
         /// </summary>
         public DataViewSchema InputSchema => ((IInputSchema)_owner).InputSchema;
         /// <summary>
         /// Numero massimo di tentativi di training del modello
         /// </summary>
         public int MaxTrainingCycles => 1;
         /// <summary>
         /// Storage del modello
         /// </summary>
         public IModelStorage ModelStorage => ((IModelStorageProvider)_owner).ModelStorage;
         /// <summary>
         /// Trainer del modello
         /// </summary>
         public IModelTrainer ModelTrainer { get; } = new ModelTrainerAuto() { MaxTrainingCycles = 1 };
         /// <summary>
         /// Nome del modello
         /// </summary>
         public string ModelName => _owner.Name;
         /// <summary>
         /// Schema di output
         /// </summary>
         private DataViewSchema OutputSchema { get; }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="contextProvider">Provider di contesto di machine learning</param>
         internal Model(ObjectDetection owner, IContextProvider<TFContext> contextProvider = default) : base(contextProvider)
         {
            _owner = owner;
            OutputSchema = DataViewSchemaBuilder.Build(
               ("ImagePath", typeof(string)),
               ("ObjectId", typeof(int)),
               ("ObjectName", typeof(string)),
               ("ObjectDisplayName", typeof(string)),
               ("Score", typeof(float)),
               ("ObjectLeft", typeof(float)),
               ("ObjectTop", typeof(float)),
               ("ObjectWidth", typeof(float)),
               ("ObjectHeight", typeof(float)));
         }
         /// <summary>
         /// Funzione di dispose
         /// </summary>
         /// <param name="disposing">Se true indica che il dispose dell'oggetto e' stato chiamato da programma</param>
         protected override void Dispose(bool disposing)
         {
            base.Dispose(disposing);
            var trainProcess = default(Process);
            lock (this)
               trainProcess = _trainProcess;
            if (trainProcess != null) {
               var taskKill = Task.Run(() =>
               {
                  _trainProcess = null;
                  try {
                     trainProcess.Kill(true);
                  }
                  catch (Exception) {
                  }
               });
               if (disposing)
                  taskKill.WaitSync();
            }
         }
         /// <summary>
         /// Restituisce il modello sottoposto al training
         /// </summary>
         /// <param name="trainer">Il trainer da utilizzare</param>
         /// <param name="data">Dati di training</param>
         /// <param name="metrics">Eventuale metrica</param>
         /// <param name="cancellation">Token di cancellazione</param>
         /// <returns>Il trasnformer di dati</returns>
         protected override IDataTransformer GetTrainedModel(IModelTrainer trainer, IDataAccess data, out object metrics, CancellationToken cancellation)
         {
            // Processo di training
            var trainProcess = default(Process);
            try {
               // Crea ed avvia i processo di training
               lock (this)
                  trainProcess = _trainProcess = new Process();
               trainProcess.StartInfo.FileName = "ODModelBuilderTF.exe";
               trainProcess.StartInfo.Arguments = @"--model_dir S:\ML.NET\MachineLearningStudio\ODModelBuilderTF\trained-model";
               trainProcess.StartInfo.Arguments += @" --train_images_dir S:\ML.NET\MachineLearningStudio\ODModelBuilderTF\images\train";
               trainProcess.StartInfo.Arguments += @" --eval_images_dir S:\ML.NET\MachineLearningStudio\ODModelBuilderTF\images\eval";
               trainProcess.StartInfo.Arguments += @" --num_train_steps 5000";
               trainProcess.StartInfo.Arguments += @" --batch_size 8";
               trainProcess.StartInfo.UseShellExecute = false;
               trainProcess.StartInfo.RedirectStandardOutput = true;
               trainProcess.StartInfo.RedirectStandardError = true;
               trainProcess.StartInfo.CreateNoWindow = true;
               trainProcess.Start();
               // Task di ascolto
               Task.Run(() =>
               {
                  // Ascolto sullo standard output
                  var taskLogOutput = Task.Run(() =>
                  {
                     try {
                        for (var line = trainProcess.StandardOutput.ReadLine(); line != null && !cancellation.IsCancellationRequested; line = trainProcess.StandardOutput.ReadLine())
                           Channel.WriteLog(line);
                     }
                     catch (Exception) { }
                  });
                  // Ascolto sullo standard error
                  var taskLogError = Task.Run(() =>
                  {
                     try {
                        for (var line = trainProcess.StandardError.ReadLine(); line != null && !cancellation.IsCancellationRequested; line = trainProcess.StandardError.ReadLine())
                           Channel.WriteLog(line);
                     }
                     catch (Exception) { }
                  });
                  // Attesa fine training o cancellazione
                  while (!taskLogOutput.IsCompleted && !taskLogError.IsCompleted && !cancellation.IsCancellationRequested)
                     Thread.Sleep(1000);
                  // Effettua il kill del processo se richiasta cancellazione
                  if (cancellation.IsCancellationRequested) {
                     try {
                        try {
                           trainProcess.CancelOutputRead();
                           taskLogOutput.WaitSync();
                        }
                        catch (Exception) { }
                        try {
                           trainProcess.CancelErrorRead();
                           taskLogError.WaitSync();
                        }
                        catch (Exception) { }
                        trainProcess.Kill(true);
                     }
                     catch (Exception exc) {
                        Trace.WriteLine(exc);
                     }
                  }
                  // Attende termine processo
                  trainProcess.WaitForExit();
               }).WaitSync();
               metrics = null;
               return this;
            }
            finally {
               // Finelizzazione operazioni
               if (trainProcess != null)
                  trainProcess.Dispose();
               lock (this)
                  _trainProcess = null;
            }
         }
         /// <summary>
         /// Trasforma i dati di input per il modello
         /// </summary>
         /// <param name="data">Dati di input</param>
         /// <param name="cancellation">Eventuale token di cancellazione</param>
         /// <returns>I dati trasformati</returns>
         IDataAccess IDataTransformer.Transform(IDataAccess data, CancellationToken cancellation)
         {
            // Crea la sessione di trasformazione
            cancellation.ThrowIfCancellationRequested();
            // Prepara la configurazione della rete neurale
            // Dati di input e di output
            var input = data.ToDataViewGrid();
            var output = DataViewGrid.Create(this, OutputSchema);
            // Loop su tutte le righe di input
            foreach (var row in data.GetRowCursor(data.Schema).AsEnumerable()) {
               // Path dell'immagine
               var imagePath = (string)row.ToDataViewValuesRow(this)["ImagePath"];
               // Trasforma l'immagine in dati numerici
               var imageTensor = ReadTensorFromImageFile(imagePath);
               // La passa per la rete neurale
               cancellation.ThrowIfCancellationRequested();
               var graph = _session.as_default().graph.as_default();
               // Tensore di input
               Tensor inputTensor;
               var mobilenetV2 = false;
               try {
                  // Tensore standard
                  inputTensor = graph.OperationByName("serving_default_input_tensor").outputs[0];
               }
               catch (Exception) {
                  // Tensore per la CenterNet MobileNetV2 FPN 512x512 che hanno deciso di definire diversamente...
                  inputTensor = graph.OperationByName("serving_default_input").outputs[0];
                  mobilenetV2 = true;
               }
               var outputTensors = graph.OperationByName("StatefulPartitionedCall").outputs;
               Tensor numDetectionsTensor, boxesTensor, scoresTensor, classesTensor;
               int startId = 1;
               // Modello tipo Mask R-CNN Inception ResNet V2
               if (outputTensors.Length == 23) {
                  numDetectionsTensor = outputTensors[12];
                  boxesTensor = outputTensors[4];
                  scoresTensor = outputTensors[8];
                  classesTensor = outputTensors[5];
               }
               // Centernet ResNet / Centernet HourGlass
               else if (outputTensors.Length == 4) {
                  if (!mobilenetV2) {
                     numDetectionsTensor = outputTensors[3];
                     boxesTensor = outputTensors[0];
                     scoresTensor = outputTensors[2];
                     classesTensor = outputTensors[1];
                  }
                  else {
                     numDetectionsTensor = outputTensors[0];
                     boxesTensor = outputTensors[3];
                     scoresTensor = outputTensors[1];
                     classesTensor = outputTensors[2];
                  }
               }
               // EfficientDet / SSD / Faster R-CCN
               else if (outputTensors.Length == 8) {
                  numDetectionsTensor = outputTensors[5];
                  boxesTensor = outputTensors[1];
                  scoresTensor = outputTensors[4];
                  classesTensor = outputTensors[2];
               }
               else
                  throw new Exception("Can't infer the model type");
               var outTensorArr = new Tensor[] { numDetectionsTensor, boxesTensor, scoresTensor, classesTensor };
               var results = _session.as_default().run(outTensorArr, new FeedItem(inputTensor, imageTensor));
               // Ottiene i risultati
               cancellation.ThrowIfCancellationRequested();
               var scores = results[2].AsIterator<float>().ToArray();
               var boxes = results[1].GetData<float>().ToArray();
               var ids = np.squeeze(results[3]).GetData<float>().ToArray();
               // Riempe la vista di dati di output con le rilevazioni
               for (var i = 0; i < scores.Length; i++) {
                  var score = scores[i];
                  if (score < _minimumScore)
                     continue;
                  var label = _labels.Items.Where(w => w.id == ids[i] + (1 - startId)).FirstOrDefault();
                  if (label == default)
                     continue;
                  output.Add(
                     ("ImagePath", imagePath),
                     ("ObjectId", label.id),
                     ("ObjectName", label.name),
                     ("ObjectDisplayName", label.display_name),
                     ("Score", score),
                     ("ObjectLeft", boxes[i * 4 + 1]),
                     ("ObjectTop", boxes[i * 4]),
                     ("ObjectWidth", boxes[i * 4 + 3] - boxes[i * 4 + 1]),
                     ("ObjectHeight", boxes[i * 4 + 2] - boxes[i * 4]));
               }
            }
            return output;
         }
         /// <summary>
         /// Carica i dati da uno storage
         /// </summary>
         /// <param name="dataStorage">Storage di dati</param>
         /// <returns>La vista di dati</returns>
         public override IDataAccess LoadData(IDataStorage dataStorage) => DataViewGrid.Create(this, InputSchema);
         /// <summary>
         /// Carica il modello da uno storage
         /// </summary>
         /// <param name="modelStorage">Storage del modello</param>
         /// <param name="schema">Lo schema del modello</param>
         /// <returns>Il modello</returns>
         public override IDataTransformer LoadModel(IModelStorage modelStorage, out DataViewSchema schema)
         {
            // Disalloca l'eventuale modello precedente
            if (_session != null)
               _session.Dispose();
            // Carica modello e labels
            _session = Session.LoadFromSavedModel(Path.Combine((ModelStorage as ModelStorageFile).FilePath, "saved_model"));
            _labels = PbtxtParser.ParsePbtxtFile(Path.Combine((ModelStorage as ModelStorageFile).FilePath, "label_map.pbtxt"));
            schema = InputSchema;
            return this;
         }
         /// <summary>
         /// Legge un immagine e ne crea il tensore
         /// </summary>
         /// <param name="filePath">Path del file</param>
         /// <returns>Il tensore</returns>
         private NDArray ReadTensorFromImageFile(string filePath)
         {
            using var graph = Context.Graph().as_default();
            Context.compat.v1.disable_eager_execution();
            using var file_reader = Context.io.read_file(filePath, "file_reader");
            using var decodeJpeg = Context.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            using var casted = Context.cast(decodeJpeg, TF_DataType.TF_UINT8);
            using var dims_expander = Context.expand_dims(casted, 0);
            Context.enable_eager_execution();
            using var sess = Context.Session(graph).as_default();
            return sess.run(dims_expander);
         }
         /// <summary>
         /// Salva i dati in uno storage
         /// </summary>
         /// <param name="dataStorage">Storage di dati</param>
         /// <param name="data">Dati</param>
         public override void SaveData(IDataStorage dataStorage, IDataAccess data) { }
         /// <summary>
         /// Salva il modello in uno storage
         /// </summary>
         /// <param name="modelStorage">Storage del modello</param>
         /// <param name="model">Modello</param>
         /// <param name="schema">Lo schema del modello</param>
         public override void SaveModel(IModelStorage modelStorage, IDataTransformer model, DataViewSchema schema) { }
         #endregion
      }
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class ObjectDetection // Prediction
   {
      [Serializable]
      public class Prediction
      {
         #region class Box
         /// <summary>
         /// Box contenitivo
         /// </summary>
         public class Box
         {
            #region Properties
            /// <summary>
            /// Eventuale definizione di nome da visualizzare
            /// </summary>
            public string DisplayedName { get; }
            /// <summary>
            /// Nome oggetto. Prioritario il Displayed name, altrimenti il kind.
            /// </summary>
            public string Name => !string.IsNullOrEmpty(DisplayedName) ? DisplayedName : Kind;
            /// <summary>
            /// Altezza
            /// </summary>
            public float Height { get; }
            /// <summary>
            /// Identificatore
            /// </summary>
            public int Id { get; }
            /// <summary>
            /// Lato sinistro
            /// </summary>
            public float Left { get; }
            /// <summary>
            /// Tipo di oggetto
            /// </summary>
            public string Kind { get; }
            /// <summary>
            /// Punteggio
            /// </summary>
            public float Score { get; }
            /// <summary>
            /// Lato superiore
            /// </summary>
            public float Top { get; }
            /// <summary>
            /// Larghezza
            /// </summary>
            public float Width { get; }
            #endregion
            #region Methods
            /// <summary>
            /// Costruttore
            /// </summary>
            /// <param name="id">Identificatore oggetto</param>
            /// <param name="kind">Tipo di oggetto</param>
            /// <param name="displayedName">Eventuale definizione di nome da visualizzare</param>
            /// <param name="score">Punteggio</param>
            /// <param name="left">Lato sinistro</param>
            /// <param name="top">Lato superiore</param>
            /// <param name="width">Larghezza</param>
            /// <param name="height">Altezza</param>
            public Box(int id, string kind, string displayedName, float score, float left, float top, float width, float height)
            {
               Id = id;
               DisplayedName = displayedName;
               Height = height;
               Left = left;
               Kind = kind;
               Score = score;
               Top = top;
               Width = width;
            }
            #endregion
         }
         #endregion
         #region Properties
         /// <summary>
         /// Box di contenimento oggetti
         /// </summary>
         public Box[] Boxes { get; }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="data">Dati della previsione</param>
         internal Prediction(IDataAccess data)
         {
            var grid = data.ToDataViewGrid();
            Boxes = new Box[grid.Rows.Count];
            for (var i = 0; i < Boxes.Length; i++) {
               var row = grid.Rows[i];
               Boxes[i] = new Box(
                  row["ObjectId"],
                  row["ObjectName"],
                  row["ObjectDisplayName"],
                  row["Score"],
                  row["ObjectLeft"],
                  row["ObjectTop"],
                  row["ObjectWidth"],
                  row["ObjectHeight"]);
            }
         }
         #endregion
      }
   }
}
