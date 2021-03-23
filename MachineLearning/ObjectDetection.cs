using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.TensorFlow;
using MachineLearning.Util;
using Microsoft.ML;
using NumSharp;
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
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
         IDataTransformer,
         IInputSchema,
         IModelName,
         IModelStorageProvider
      {
         #region Fields
         /// <summary>
         /// Label degli oggetti conosciuti dal modello
         /// </summary>
         private PbtxtItems _labels;
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly ObjectDetection _owner;
         /// <summary>
         /// Sessione TensorFlow
         /// </summary>
         private Session _session;
         /// <summary>
         /// Soglia punteggio minimo
         /// </summary>
         private const float minimumScore = 0.5f;
         #endregion
         #region Properties
         /// <summary>
         /// Schema di input
         /// </summary>
         public DataViewSchema InputSchema => ((IInputSchema)_owner).InputSchema;
         /// <summary>
         /// Storage del modello
         /// </summary>
         public IModelStorage ModelStorage => ((IModelStorageProvider)_owner).ModelStorage;
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
               var imgArr = ReadTensorFromImageFile(imagePath);
               // La passa per la rete neurale
               cancellation.ThrowIfCancellationRequested();
               var graph = _session.as_default().graph.as_default();
               var t0 = graph.OperationByName("StatefulPartitionedCall").outputs;
               var tensorNum = t0[5];
               var tensorBoxes = t0[1];
               var tensorScores = t0[4];
               var tensorClasses = t0[2];
               var imgTensor = graph.OperationByName("serving_default_input_tensor").outputs[0];
               var outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };
               var results = _session.as_default().run(outTensorArr, new FeedItem(imgTensor, imgArr));
               // Ottiene i risultati
               cancellation.ThrowIfCancellationRequested();
               var scores = results[2].AsIterator<float>();
               var boxes = results[1].GetData<float>();
               var ids = np.squeeze(results[3]).GetData<float>();
               // Riempe la vista di dati di output con le rilevazioni
               for (var i = 0; i < scores.size; i++) {
                  var score = scores.MoveNext();
                  if (score < minimumScore)
                     continue;
                  var label = _labels.Items.Where(w => w.id == ids[i]).FirstOrDefault();
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
         public override IDataAccess LoadData(IDataStorage dataStorage) => null;
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
