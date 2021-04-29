using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
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
      /// <returns>Il tipo di immagine</returns>
      public Prediction GetPrediction(string imagePath) => GetPredictionAsync(imagePath, default).WaitSync();
      /// <summary>
      /// Restituisce il tipo di immagine
      /// </summary>
      /// <param name="imagePath">Path dell'immagine</param>
      /// <param name="cancel">Eventuale token di cancellazione</param>
      /// <returns>Il task di previsione del tipo di immagine</returns>
      public async Task<Prediction> GetPredictionAsync(string imagePath, CancellationToken cancel = default) =>
         new Prediction(_model, await _model.GetPredictionDataAsync(new[] { imagePath }, cancel));
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione del training</param>
      public Task StartTrainingAsync(CancellationToken cancellation = default) => _model.StartTrainingAsync(cancellation);
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione dell'attesa</param>
      public Task StopTrainingAsync() => _model.StopTrainingAsync();
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
                  .Append(Context.Transforms.ResizeImages(
                     inputColumnName: "Image",
                     outputColumnName: "ResizedImage",
                     imageWidth: Config.ImageSize.Width,
                     imageHeight: Config.ImageSize.Height,
                     resizing: ImageResizingEstimator.ResizingKind.Fill))
                  .Append(Context.Transforms.ExtractPixels(
                     inputColumnName: "ResizedImage",
                     outputColumnName: Config.Inputs[0].ColumnName,
                     interleavePixelColors: true,
                     outputAsFloatArray: Config.Inputs[0].DataType == typeof(float))),
               Trainer =
                  Config.Format switch
                  {
                     ODModelConfig.ModelFormat.Onnx =>
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
                     ODModelConfig.ModelFormat.TF2SavedModel =>
                        Context.Model.LoadTensorFlowModel(Path.GetDirectoryName(Config.ModelFilePath)).ScoreTensorFlowModel(
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
            var config = ODModelConfig.Load(modelStorage.ImportPath);
            // Verifica se il modello e' di tipo noto
            if (config.Format != ODModelConfig.ModelFormat.Onnx && config.Format != ODModelConfig.ModelFormat.TF2SavedModel)
               return null;
            // Memorizza la configurazione
            Config = config;
            // Verifica se il modello ML.NET e' piu' recente del modello da importare
            if (modelStorage is IDataTimestamp modelTimestamp && modelTimestamp.DataTimestamp >= File.GetLastWriteTime(config.ModelFilePath))
               return null;
            // Ottiene le pipes
            var pipelines = GetPipes();
            // Crea il modello
            var dataView = DataViewGrid.Create(this, InputSchema);
            var model = pipelines.Merged.Fit(dataView);
            schema = InputSchema;
            var result = new DataTransformerMLNet(this, model);
            // Salva modello. Non per il Tensorflow saved_model: baco della ML.NET nel salvataggio.
            if (config.Format != ODModelConfig.ModelFormat.TF2SavedModel)
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
         /// Dizionario di trasformazione da nome colonna a indice
         /// </summary>
         private static Dictionary<string, int> columnToIndex;
         #endregion
         #region Properties
         /// <summary>
         /// Labels
         /// </summary>
         public ReadOnlyCollection<string> Labels { get; }
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
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="data">Dati della previsione</param>
         internal Prediction(Model owner, IDataAccess data)
         {
            var grid = data.ToDataViewGrid();
            Labels = owner.Labels;
            DetectionBoxes = grid[0]["detection_boxes"];
            DetectionClasses = grid[0]["detection_classes"];
            DetectionScores = grid[0]["detection_scores"];
            //if (columnToIndex == null) {
            //   var meaning = new TextMeaningRecognizer(owner)
            //   {
            //      DataStorage = new DataStorageBinaryMemory()
            //   };
            //   var texts = DataViewGrid.Create(owner, meaning.InputSchema);
            //   texts.Add(nameof(DetectionBoxes), "detection boxes");
            //   texts.Add(nameof(DetectionBoxes), "detection_boxes");
            //   texts.Add(nameof(DetectionBoxes), "detected boxes");
            //   texts.Add(nameof(DetectionBoxes), "boxes");
            //   texts.Add(nameof(DetectionClasses), "detection classes");
            //   texts.Add(nameof(DetectionClasses), "detection_classes");
            //   texts.Add(nameof(DetectionClasses), "detected classes");
            //   texts.Add(nameof(DetectionClasses), "classes");
            //   texts.Add(nameof(DetectionScores), "detection scores");
            //   texts.Add(nameof(DetectionScores), "detection_scores");
            //   texts.Add(nameof(DetectionScores), "detected_scores");
            //   texts.Add(nameof(DetectionScores), "scores");
            //   meaning.DataStorage.SaveData(owner.Context, texts);

            //   meaning.GetPredictionAsync(default, "detection_scores").WaitSync();

            //   var meanings =
            //      (from c in grid.Schema
            //       select new { Meaning = meaning.GetPrediction(c.Name), c.Index }).ToArray();
            //   columnToIndex = new();
            //   columnToIndex[nameof(DetectionBoxes)] = meanings.OrderByDescending(m => m.Meaning.Score).First().Index;
            //}

            //Kind = grid[0]["PredictedLabel"];
            //var scores = (float[])grid[0]["Score"];
            //var slotNames = grid.Schema["Score"].GetSlotNames();
            //Scores = slotNames.Zip(scores).Select(item => new KeyValuePair<string, float>(item.First, item.Second)).ToArray();
            //Score = Scores.FirstOrDefault(s => s.Key == Kind).Value;
         }
         /// <summary>
         /// Restituisce i bounding box filtrati
         /// </summary>
         /// <param name="minScore">Punteggio minimo (0 ... 1)</param>
         /// <returns>La lista di bounding box</returns>
         public List<Box> GetBoxes(double minScore = 0.75)
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
