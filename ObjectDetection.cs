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
         /// Modello per ricavare il nome di una colonna dal nome della proprieta' a cui accedere
         /// </summary>
         private static Dictionary<string, int> columnIndex;
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
            // Crea la griglia del risultato
            var grid = data.ToDataViewGrid();
            // Memorizza le labels
            Labels = owner.Labels;
            // Crea il dizionario intelligente di mappatura degli ingressi uscite
            if (columnIndex == null) {
               // Modello di interpretazione del testo in base ai nomi delle colonne del modello
               var ml = owner.Context;
               var pipe =
                  ml.Transforms.Text.NormalizeText("Text")
                  .Append(ml.Transforms.Text.TokenizeIntoWords("Text", null, new[] { '_', ',', ':', '[', ']', ' ' }))
                  .Append(ml.Transforms.Text.FeaturizeText("Text", "Text"))
                  .Append(ml.Transforms.Conversion.MapValueToKey("ColumnName"))
                  .Append(ml.MulticlassClassification.Trainers.NaiveBayes("ColumnName", "Text"))
                  .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
                  ;
               var inputOutputs = ml.Data.LoadFromEnumerable(from g in new[] { owner.Config.Inputs, owner.Config.Outputs }
                                                             from c in g
                                                             select new { ColumnName = c.Name, Text = c.Name });
               var model = pipe.Fit(inputOutputs);
               // Predice il nome della colonna che assomiglia di piu' all'oggetto richiesto
               var predictions = DataViewGrid.Create(new DataAccess(ml, model.Transform(ml.Data.LoadFromEnumerable(new[]
               {
                  new { ColumnName = "", Text = "boxes" },
                  new { ColumnName = "", Text = "classes" },
                  new { ColumnName = "", Text = "scores" },
               }))));
               // Crea il dizionario di corrispondenze proprieta' / indice colonna
               columnIndex = new()
               {
                  { nameof(DetectionBoxes), grid.Schema[(string)predictions[0]["PredictedLabel"]].Index },
                  { nameof(DetectionClasses), grid.Schema[(string)predictions[1]["PredictedLabel"]].Index },
                  { nameof(DetectionScores), grid.Schema[(string)predictions[2]["PredictedLabel"]].Index },
               };
            }
            // Memorizza i risultati della previsione
            DetectionBoxes = grid[0][columnIndex[nameof(DetectionBoxes)]];
            DetectionClasses = grid[0][columnIndex[nameof(DetectionClasses)]];
            DetectionScores = grid[0][columnIndex[nameof(DetectionScores)]];
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
