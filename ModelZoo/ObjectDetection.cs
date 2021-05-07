using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;

namespace MachineLearning.ModelZoo
{
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
            // Crea la pipeline di output
            var outputEstimators = new List<IEstimator<ITransformer>>();
            var columnNameTransform = (
               from g in new[] { Config.Inputs, Config.Outputs }
               from c in g
               where c.Name != c.ColumnName
               select c).ToArray();
            foreach (var c in columnNameTransform)
               outputEstimators.Add(Context.Transforms.CopyColumns(inputColumnName: c.ColumnName, outputColumnName: c.Name));
            if (columnNameTransform.Length > 0)
               outputEstimators.Add(Context.Transforms.DropColumns((from c in columnNameTransform where c.Name != c.ColumnName select c.ColumnName).ToArray()));
            var outputPipeline = outputEstimators.Count < 1 ? null : outputEstimators.Count > 1 ? outputEstimators[0].Append(outputEstimators[1]) : outputEstimators[0];
            for (var i = 2; i < outputEstimators.Count; i++)
               outputPipeline = outputPipeline.Append(outputEstimators[i]);
            // Restituisce le tre pipe di learning
            return _pipes ??= new()
            {
               Input =
                  Context.Transforms.LoadImages(
                     inputColumnName: "ImagePath",
                     outputColumnName: "Image",
                     imageFolder: "")
                  .Append(Context.Transforms.Expression(inputColumnNames: new[] { "ImagePath" }, outputColumnName: "ModelWidth", expression: $"w => {Config.ImageSize.Width}f"))
                  .Append(Context.Transforms.Expression(inputColumnNames: new[] { "ImagePath" }, outputColumnName: "ModelHeight", expression: $"w => {Config.ImageSize.Height}f"))
                  .Append(Context.Transforms.ResizeImages(
                     inputColumnName: "Image",
                     outputColumnName: "ResizedImage",
                     imageWidth: Config.ImageSize.Width,
                     imageHeight: Config.ImageSize.Height,
                     resizing: ImageResizingEstimator.ResizingKind.Fill))
                  .Append(Context.Transforms.ExtractPixels(
                     inputColumnName: "ResizedImage",
                     outputColumnName: Config.Inputs[0].ColumnName,
                     scaleImage: !Config.ModelType.ToLower().Contains("yolo") ? 1f : 1f /255f,
                     interleavePixelColors: !Config.ModelType.ToLower().Contains("yolo"),
                     outputAsFloatArray: Config.Inputs[0].DataType == typeof(float))),
               Trainer =
                  Config.Format switch
                  {
                     // Onnx TensorFlow
                     var tf when tf == ODModelConfig.ModelFormat.Onnx && !Config.ModelType.ToLower().Contains("yolo") =>
                        Context.Transforms.ApplyOnnxModel(
                           inputColumnNames: new[] { Config.Inputs[0].ColumnName },
                           outputColumnNames: (from c in Config.Outputs select c.ColumnName).ToArray(),
                           modelFile: Config.ModelFilePath,
                           shapeDictionary: new Dictionary<string, int[]>()
                           {
                              {
                                 Config.Inputs[0].ColumnName,
                                 Config.Inputs[0].Dim.Take(1).Concat(new[] { Config.ImageSize.Width, Config.ImageSize.Height }).Concat(Config.Inputs[0].Dim.Skip(3).Take(1)).ToArray()
                              }
                           }),
                     // Onnx Yolo
                     var tf when tf == ODModelConfig.ModelFormat.Onnx && Config.ModelType.ToLower().Contains("yolo") =>
                        new OnnxYolov5Estimator(
                           Context,
                           inputColumnName: Config.Inputs[0].ColumnName,
                           outputColumnNames: (from c in Config.Outputs select c.ColumnName).ToArray(),
                           outputShapes: (from c in Config.Outputs select c.Dim).ToArray(),
                           modelFile: Config.ModelFilePath),
                     // saved_model o frozen graph TensorFlow
                     var tf when tf == ODModelConfig.ModelFormat.TF2SavedModel || tf == ODModelConfig.ModelFormat.TFFrozenGraph =>
                        Context.Model.LoadTensorFlowModel(Config.ModelFilePath).ScoreTensorFlowModel(
                           inputColumnNames: new[] { Config.Inputs[0].ColumnName },
                           outputColumnNames: (from c in Config.Outputs select c.ColumnName).ToArray()),
                     _ => throw new FormatException("Unknown model format")
                  },
               Output = outputPipeline
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
   [CustomMappingFactoryAttribute(nameof(OnnxYolov5Estimator))]
   internal class OnnxYolov5Estimator : CustomMappingFactory<OnnxYolov5Estimator.Input, OnnxYolov5Estimator.Output>, IEstimator<OnnxYolov5Estimator>, ITransformer
   {
      #region Fields
      /// <summary>
      /// La pipeline
      /// </summary>
      IEstimator<ITransformer> pipeline;
      /// <summary>
      /// Il transformer
      /// </summary>
      ITransformer transformer;
      #endregion
      #region Properties
      /// <summary>
      /// Indicatore di RowToRowMapper
      /// </summary>
      public bool IsRowToRowMapper => true;
      #endregion
      /// <summary>
      /// Costruttore
      /// </summary>
      public OnnxYolov5Estimator() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="outputColumnNames">Nomi delle colonne di output</param>
      /// <param name="inputColumnName">Nome della colonna di input</param>
      /// <param name="outputShapes"></param>
      /// <param name="modelFile"></param>
      public OnnxYolov5Estimator(MLContext ml, string[] outputColumnNames, int[][] outputShapes, string inputColumnName, string modelFile)
      {
         // Elenco di estimatore
         var estimators = new List<IEstimator<ITransformer>>();
         // Aggiunge il modello onnx
         estimators.Add(
            ml.Transforms.ApplyOnnxModel(
               inputColumnNames: new[] { inputColumnName },
               outputColumnNames: outputColumnNames,
               modelFile: modelFile));
         // Copia la colonna di output nel input del custom mapping
         if (outputShapes != null) {
            // Sceglie la l'output del modello piu' grande, quello con tutti gli anchors concatenati
            var (Name, Shape) =
               outputColumnNames
               .Zip(outputShapes)
               .Select(item => (Name: item.First, Shape: item.Second))
               .OrderBy(item => { var m = 1; item.Shape.ToList().ForEach(s => m *= s); return m; })
               .Last();
            estimators.Add(ml.Transforms.CopyColumns(nameof(Input.Detections), Name));
         }
         // Se non specificate le shapes, sceglie il primo output
         else
            estimators.Add(ml.Transforms.CopyColumns(nameof(Input.Detections), outputColumnNames[0]));
         // Aggiunge il custom mapping per l'interpretazione dell'output del modello, la conversione in previsione standard e il NMS
         estimators.Add(ml.Transforms.CustomMapping(GetMapping(), nameof(OnnxYolov5Estimator)));
         // Elimina le colonne temporanee
         estimators.Add(ml.Transforms.DropColumns((from c in new[] { nameof(Input.Detections) }.Concat(outputColumnNames) select c).ToArray()));
         // Costruisce la pipe
         pipeline = estimators.Count < 1 ? null : estimators.Count > 1 ? estimators[0].Append(estimators[1]) : estimators[0];
         for (var i = 2; i < estimators.Count; i++)
            pipeline = pipeline.Append(estimators[i]);
      }
      /// <summary>
      /// Implementazione della fit
      /// </summary>
      /// <param name="input">Dati di input</param>
      /// <returns>l'Estimator</returns>
      public OnnxYolov5Estimator Fit(IDataView input)
      {
         transformer = pipeline.Fit(input);
         return this;
      }
      /// <summary>
      /// Restituisce la shape dello schema di output dell'estimator
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>La shape dello schema</returns>
      public SchemaShape GetOutputSchema(SchemaShape inputSchema) => pipeline.GetOutputSchema(inputSchema);
      /// <summary>
      /// Restituisce lo schema di output dei dati
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Lo schema di output</returns>
      public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => transformer.GetOutputSchema(inputSchema);
      /// <summary>
      /// Implementazione della transform
      /// </summary>
      /// <param name="input">Dati di input</param>
      /// <returns>I dati trafromati</returns>
      public IDataView Transform(IDataView input) => transformer.Transform(input);
      /// <summary>
      /// Restituisce il row to row mapper
      /// </summary>
      /// <param name="inputSchema">Schema dei dati d input</param>
      /// <returns>L'interfaccia del mappatore</returns>
      public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => transformer.GetRowToRowMapper(inputSchema);
      /// <summary>
      /// Implementazione della save
      /// </summary>
      /// <param name="ctx">Contesto di salvataggio</param>
      public void Save(ModelSaveContext ctx) => transformer.Save(ctx);
      /// <summary>
      /// Azione di rimappatura
      /// </summary>
      /// <returns>L'azione</returns>
      public override Action<Input, Output> GetMapping() => new((In, Out) =>
      {
         var numLabels = 1; //@@@
         var characteristics = 5 + numLabels; // @@@
         var scoreConfidence = 0.2f;
         var perCategoryConfidence = 0.25f;
         var results = new List<(RectangleF Box, float Class, float Score, bool Valid)>();
         //var boxes = new List<RectangleF>();
         //var classes = new List<float>();
         //var scores = new List<float>();
         for (int i = 0; i < In.Detections.Length; i += characteristics) {
            // Get offset in float array
            int offset = characteristics * i;
            // Filtra alcuni box
            var objConf = In.Detections[i + 4];
            if (objConf <= scoreConfidence)
               continue;
            // Ottiene il punteggio reale della classe
            var classProbs = new List<float>();
            for (var j = 0; j < numLabels; j++)
               classProbs.Add(In.Detections[i + 5 + j]);
            var allScores = classProbs.Select(p => p * objConf).ToList();
            // Ottiene la miglior classe ed indice
            float maxConf = allScores.Max();
            if (maxConf < perCategoryConfidence)
               continue;
            float maxClass = allScores.ToList().IndexOf(maxConf);
            // Aggiunge il risultato
            var x1 = (In.Detections[i + 1] - In.Detections[i + 3] / 2) / In.ModelWidth; //top left x
            var y1 = (In.Detections[i + 0] - In.Detections[i + 2] / 2) / In.ModelHeight; //top left y
            var x2 = (In.Detections[i + 1] + In.Detections[i + 3] / 2) / In.ModelWidth; //bottom right x
            var y2 = (In.Detections[i + 0] + In.Detections[i + 2] / 2) / In.ModelHeight; //bottom right y
            results.Add((Box: new RectangleF(x1, y1, x2 - x1, y2 - y1), Class: maxClass + 1, Score: maxConf, Valid: true));
         }
         // NMS
         var nmsOverlapRatio = 0.45f;
         for (var i = 0; i < results.Count; i++) {
            var item = results[i];
            if (!item.Valid)
               continue;
            for (var j = 0; j < results.Count; j++) {
               var current = results[j];
               if (current == item || !current.Valid)
                  continue;
               var intersection = RectangleF.Intersect(item.Box, current.Box);
               var intArea = intersection.Width * intersection.Height;
               var unionArea = item.Box.Width * item.Box.Height + current.Box.Width * current.Box.Height - intArea;
               var overlap = intArea / unionArea;
               if (overlap > nmsOverlapRatio) {
                  if (item.Score > current.Score) {
                     current.Valid = false;
                     results[j] = current;
                  }
               }
            }
         }
         results = results.Where(item => item.Valid).ToList();
         Out.Boxes = new float[results.Count * 4];
         Out.Classes = new float[results.Count];
         Out.Scores = new float[results.Count];
         for (var i = 0; i < results.Count; i++) {
            var (Box, Class, Score, Valid) = results[i];
            var iBox = i * 4;
            Out.Boxes[iBox + 0] = Box.Left;
            Out.Boxes[iBox + 1] = Box.Top;
            Out.Boxes[iBox + 2] = Box.Right;
            Out.Boxes[iBox + 3] = Box.Bottom;
            Out.Classes[i] = Class;
            Out.Scores[i] = Score;
         }
      });
      /// <summary>
      /// Classe di dati di input del mappatore
      /// </summary>
      internal class Input
      {
         #region Properties
         /// <summary>
         /// Array di rilevamenti
         /// </summary>
         public float[] Detections { get; set; }
         /// <summary>
         /// Altezza del modello
         /// </summary>
         public float ModelHeight { get; set; }
         /// <summary>
         /// Larghezza del modello
         /// </summary>
         public float ModelWidth { get; set; }
         #endregion
      }
      /// <summary>
      /// Classe di dati di output del mappatore
      /// </summary>
      internal class Output
      {
         #region Properties
         /// <summary>
         /// Classi di rilevamento
         /// </summary>
         [ColumnName("detection_classes")]
         public float[] Classes { get; set; }
         /// <summary>
         /// Punteggi di rilevamento
         /// </summary>
         [ColumnName("detection_scores")]
         public float[] Scores { get; set; }
         /// <summary>
         /// Box di rilevamento
         /// </summary>
         [ColumnName("detection_boxes")]
         public float[] Boxes { get; set; }
         #endregion
      }
   }
}
