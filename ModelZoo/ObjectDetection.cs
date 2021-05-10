using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Transforms;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;

namespace MachineLearning.ModelZoo
{
   [Serializable]
   public sealed partial class ObjectDetection :
      IInputSchema,
      IModelStorageProvider,
      IModelTrainingControl
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
      /// Schema di input dei dati
      /// </summary>
      public ReadOnlyCollection<string> Labels => _model?.Labels;
      /// <summary>
      /// Storage del modello
      /// </summary>
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
      public ObjectDetection(IContextProvider<MLContext> context = default)
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
      /// <param name="cancel">Eventuale token di cancellazione</param>
      /// <returns>I box dei rilevamenti</returns>
      public Prediction GetPrediction(string imagePath, CancellationToken cancel = default) =>
         new(_model, _model.GetPredictionData(new[] { imagePath }, cancel));
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione del training</param>
      public void StartTraining(CancellationToken cancellation = default) => _model.StartTraining(cancellation);
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione dell'attesa</param>
      public void StopTraining(CancellationToken cancellation = default) => _model.StopTraining(cancellation);
      #endregion
   }

   /// <summary>
   /// Modello
   /// </summary>
   public sealed partial class ObjectDetection // Prediction
   {
      [Serializable]
      public sealed class Model :
         ModelBaseMLNet,
         IDataStorageProvider,
         IDataTransformer,
         IInputSchema,
         IModelName,
         IModelStorageProvider
      {
         #region Fields
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly ObjectDetection _owner;
         /// <summary>
         /// Pipe di training
         /// </summary>
         [NonSerialized]
         private ModelPipes _pipes;
         #endregion
         #region Properties
         /// <summary>
         /// Configurazione del modello
         /// </summary>
         internal ODModelConfig Config { get; private set; }
         /// <summary>
         /// Storage di dati
         /// </summary>
         public IDataStorage DataStorage { get; } = new DataStorageBinaryMemory();
         /// <summary>
         /// Schema di input
         /// </summary>
         public DataViewSchema InputSchema => ((IInputSchema)_owner).InputSchema;
         /// <summary>
         /// Labels
         /// </summary>
         public ReadOnlyCollection<string> Labels => Config.Labels;
         /// <summary>
         /// Storage del modello
         /// </summary>
         public IModelStorage ModelStorage => ((IModelStorageProvider)_owner).ModelStorage;
         /// <summary>
         /// Nome del modello
         /// </summary>
         public string ModelName => _owner.Name;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="contextProvider">Provider di contesto di machine learning</param>
         internal Model(ObjectDetection owner, IContextProvider<MLContext> contextProvider = default) : base(contextProvider) => _owner = owner;
         /// <summary>
         /// Restituisce le pipe di training del modello
         /// </summary>
         /// <returns>Le pipe</returns>
         public override ModelPipes GetPipes()
         {
            // Verifica se modello yolo
            var isYolo = Config.ModelType.ToLower().Contains("yolo");
            // Crea la pipeline di output
            var outputEstimators = new EstimatorList();
            var dropColumns = new HashSet<string>();
            // Mapping dell'output dei modelli yolo
            if (isYolo) {
               // Trasforma l'output Yolov5 in output standard
               outputEstimators.Add(Context.Transforms.ScoreYolov5());
               // Rimuove le colonne di uscita del modello Yolo
               (from c in Config.Outputs select c.ColumnName).ToList().ForEach(c => dropColumns.Add(c));
            }
            // Colonne da trasformare nel caso il nome del tensore del modello non corrisponda alla colonna ML.NET
            var columnNameTransform = (
               from g in new[] { Config.Inputs, Config.Outputs }
               from c in g
               where c.Name != c.ColumnName
               select c).ToArray();
            // Rinomina le colonne
            foreach (var c in columnNameTransform)
               outputEstimators.Add(Context.Transforms.CopyColumns(inputColumnName: c.ColumnName, outputColumnName: c.Name));
            // Aggiunge le colonne da rimuovere alla fine al set
            columnNameTransform.ToList().ForEach(c => dropColumns.Add(c.ColumnName));
            dropColumns.Add("Image");
            dropColumns.Add("ResizedImage");
            Config.Inputs.ToList().ForEach(c => dropColumns.Add(c.ColumnName));
            // Rimuove le colonne
            if (dropColumns.Count > 0)
               outputEstimators.Add(Context.Transforms.DropColumns(dropColumns.ToArray()));
            // Dimensioni da inserire nel tensore di ingresso onnx nel caso manchino
            var shapes = new Queue<int>(new[] { Config.ImageSize.Width, Config.ImageSize.Height });
            // Restituisce le tre pipe di learning
            return _pipes ??= new()
            {
               // Pipe di input dati
               Input =
                  Context.Transforms.LoadImages(
                     inputColumnName: "ImagePath",
                     outputColumnName: "Image",
                     imageFolder: "")
                  .Append(Context.Transforms.ResizeImages(
                     inputColumnName: "Image",
                     outputColumnName: "ResizedImage",
                     imageWidth: Config.ImageSize.Width,
                     imageHeight: Config.ImageSize.Height,
                     resizing: ImageResizingEstimator.ResizingKind.Fill))
                  .Append(Context.Transforms.ExtractPixels(
                     inputColumnName: "ResizedImage",
                     outputColumnName: Config.Inputs[0].ColumnName,
                     scaleImage: !isYolo ? 1f : 1f / 255f,
                     interleavePixelColors: !isYolo,
                     outputAsFloatArray: Config.Inputs[0].DataType == typeof(float))),
               // Pipe di inferenza modello
               Trainer =
                  Config.Format switch
                  {
                     // Onnx
                     ODModelConfig.ModelFormat.Onnx =>
                        Context.Transforms.ApplyOnnxModel(
                           inputColumnNames: new[] { Config.Inputs[0].ColumnName },
                           outputColumnNames: (from c in Config.Outputs select c.ColumnName).ToArray(),
                           modelFile: Config.ModelFilePath,
                           shapeDictionary: new Dictionary<string, int[]>()
                           {
                              {
                                 Config.Inputs[0].ColumnName,
                                 Config.Inputs[0].Dim.Select(d => d > 0 ? d : shapes.Dequeue()).ToArray()
                              }
                           }),
                     // saved_model o frozen graph TensorFlow
                     var tf when tf == ODModelConfig.ModelFormat.TF2SavedModel || tf == ODModelConfig.ModelFormat.TFFrozenGraph =>
                        Context.Model.LoadTensorFlowModel(Config.ModelFilePath).ScoreTensorFlowModel(
                           inputColumnNames: new[] { Config.Inputs[0].ColumnName },
                           outputColumnNames: (from c in Config.Outputs select c.ColumnName).ToArray()),
                     _ => throw new FormatException("Unknown model format")
                  },
               // pipe di uscita
               Output = outputEstimators.GetPipe()
            };
         }
         /// <summary>
         /// Importa un modello esterno
         /// </summary>
         /// <param name="modelStorage">Storage del modello</param>
         /// <param name="schema">Lo schema del modello</param>
         /// <returns>Il modello</returns>
         public override IDataTransformer ImportModel(IModelStorage modelStorage, out DataViewSchema schema)
         {
            schema = null;
            // Carica la configurazione del modello
            Config = ODModelConfig.Load(modelStorage.ImportPath);
            // Verifica se il modello e' di tipo noto
            if (Config.Format == ODModelConfig.ModelFormat.Unknown)
               return null;
            // Verifica se il modello ML.NET e' piu' recente del modello da importare
            if (modelStorage is IDataTimestamp modelTimestamp && modelTimestamp.DataTimestamp >= File.GetLastWriteTimeUtc(Config.ModelFilePath))
               return null;
            // Ottiene le pipes
            var pipelines = GetPipes();
            // Crea il modello
            var dataView = DataViewGrid.Create(this, InputSchema);
            var model = pipelines.Merged.Fit(dataView);
            schema = InputSchema;
            var result = new DataTransformerMLNet(this, model);
            // Salva modello. Non per il Tensorflow saved_model: baco della ML.NET nel salvataggio.
            if (Config.Format != ODModelConfig.ModelFormat.TF2SavedModel)
               SaveModel(modelStorage, result, schema);
            return result;
         }
         #endregion
      }
   }

   /// <summary>
   /// Risultato della previsione
   /// </summary>
   public sealed partial class ObjectDetection // Prediction
   {
      [Serializable]
      public partial class Prediction
      {
         #region Fields
         /// <summary>
         /// Modello per ricavare il nome di una colonna dal nome della proprieta' a cui accedere
         /// </summary>
         private static Dictionary<string, Func<DataViewGrid, float[]>> readOutput;
         #endregion
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
         /// Costruttore
         /// </summary>
         /// <param name="data">Dati della previsione</param>
         internal Prediction(Model owner, IDataAccess data)
         {
            // Crea il dizionario intelligente di mappatura degli ingressi uscite
            if (readOutput == null) {
               var sd = new SmartDictionary<string>(from c in data.Schema select new KeyValuePair<string, string>(c.Name, c.Name));
               var columnIndex = new Dictionary<string, int>
               {
                  { nameof(DetectionBoxes), data.Schema[sd.Similar["detection boxes"]].Index },
                  { nameof(DetectionClasses), data.Schema[sd.Similar["detection classes"]].Index },
                  { nameof(DetectionScores), data.Schema[sd.Similar["detection scores"]].Index },
               };
               readOutput = new()
               {
                  { nameof(DetectionBoxes), grid => grid[0][columnIndex[nameof(DetectionBoxes)]] },
                  { nameof(DetectionClasses), grid => grid[0][columnIndex[nameof(DetectionClasses)]] },
                  { nameof(DetectionScores), grid => grid[0][columnIndex[nameof(DetectionScores)]] },
               };
            }
            // Crea la griglia del risultato
            var grid = data.ToDataViewGrid();
            // Memorizza le labels
            Labels = owner.Labels;
            // Memorizza i risultati della previsione
            ImagePath = grid[0]["ImagePath"];
            DetectionBoxes = readOutput[nameof(DetectionBoxes)](grid);
            DetectionClasses = readOutput[nameof(DetectionClasses)](grid);
            DetectionScores = readOutput[nameof(DetectionScores)](grid);
         }
         /// <summary>
         /// Restituisce i bounding box filtrati
         /// </summary>
         /// <param name="minScore">Punteggio minimo (0 ... 1)</param>
         /// <returns>La lista di bounding box</returns>
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
            /// Nome
            /// </summary>
            public string Name { get; }
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
            /// <param name="name">Nome dell'oggetto</param>
            /// <param name="score">Punteggio</param>
            /// <param name="left">Lato sinistro</param>
            /// <param name="top">Lato superiore</param>
            /// <param name="width">Larghezza</param>
            /// <param name="height">Altezza</param>
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
