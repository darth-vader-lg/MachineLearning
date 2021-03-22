using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.TensorFlow;
using MachineLearning.Util;
using Microsoft.ML;
using NumSharp;
using System;
using System.Collections.Generic;
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
      IInputSchema
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
      public string ModelStorage { get; set; }
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
      public async Task<Prediction> GetPredictionAsync(string imagePath, CancellationToken cancel = default)
      {
         await _model.GetPredictionDataAsync(new[] { imagePath }, cancel);
         return null;
      }
      #endregion
   }

   /// <summary>
   /// Modello
   /// </summary>
   public sealed partial class ObjectDetection // Prediction
   {
      [Serializable]
      public sealed class Model : ModelBaseTensorFlow, IDataTransformer
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
         private const float MIN_SCORE = 0.5f;
         #endregion
         /// <summary>
         /// Schema di output
         /// </summary>
         private DataViewSchema OutputSchema { get; }
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
               ("ObjectLeft", typeof(float)),
               ("ObjectTop", typeof(float)),
               ("ObjectWidth", typeof(float)),
               ("ObjectHeight", typeof(float)));
         }
         /// <summary>
         /// Crea l'immagine di output
         /// </summary>
         /// <param name="inputImagePath">Path dell'immagine di input</param>
         /// <param name="resultArr">Risultato della previsione</param>
         private void BuildOutputImage(string inputImagePath, NDArray[] resultArr)
         {
            // Crea il bitmap dall'immagine
            var bitmap = new Bitmap(inputImagePath);
            var scores = resultArr[2].AsIterator<float>();
            var boxes = resultArr[1].GetData<float>();
            var id = np.squeeze(resultArr[3]).GetData<float>();
            for (int i = 0; i < scores.size; i++) {
               var score = scores.MoveNext();
               if (score > MIN_SCORE) {
                  var top = boxes[i * 4] * bitmap.Height;
                  var left = boxes[i * 4 + 1] * bitmap.Width;
                  var bottom = boxes[i * 4 + 2] * bitmap.Height;
                  var right = boxes[i * 4 + 3] * bitmap.Width;
                  var rect = new Rectangle()
                  {
                     X = (int)left,
                     Y = (int)top,
                     Width = (int)(right - left),
                     Height = (int)(bottom - top)
                  };
                  var name = _labels.Items.Where(w => w.Id == id[i])
                     .Select(s => !string.IsNullOrWhiteSpace(s.DisplayName) ? s.DisplayName : s.Name)
                     .FirstOrDefault();
                  DrawObjectOnBitmap(bitmap, rect, score, name);
               }
            }
            var path = Path.ChangeExtension(inputImagePath, null) + "-labeled" + Path.GetExtension(inputImagePath);
            bitmap.Save(path);
            Console.WriteLine($"Processed image is saved as {path}");
         }
         /// <summary>
         /// Marca l'immagine
         /// </summary>
         /// <param name="bmp">Bitmap da marcare</param>
         /// <param name="rect">Rettangolo da disegnare</param>
         /// <param name="score">Punteggio della previsione</param>
         /// <param name="name">Nome della previsione</param>
         private static void DrawObjectOnBitmap(Bitmap bmp, Rectangle rect, float score, string name)
         {
            using var graphic = Graphics.FromImage(bmp); graphic.SmoothingMode = SmoothingMode.AntiAlias;
            using var pen = new Pen(Color.Lime, 2); graphic.DrawRectangle(pen, rect);
            using var font = new Font("Verdana", 8);
            var p = new Point(rect.Left + 5, rect.Top + 5);
            var text = string.Format("{0}:{1}%", name, (int)(score * 100));
            var size = graphic.MeasureString(text, font);
            using var brush = new SolidBrush(Color.FromArgb(50, Color.Lime));
            graphic.FillRectangle(brush, p.X, p.Y, size.Width, size.Height);
            graphic.DrawString(text, font, Brushes.Black, p);
         }
         /// <summary>
         /// Restituisce il task di previsione
         /// </summary>
         /// <param name="data">Riga di dati da usare per la previsione</param>
         /// <param name="cancellation">Eventule token di cancellazione attesa</param>
         /// <returns>La previsione</returns>
         public new async Task<IDataAccess> GetPredictionDataAsync(IEnumerable<object> data, CancellationToken cancellation)
         {
            //return await base.GetPredictionDataAsync(data, cancellation);
            // Importa il grafico del modello
            await Task.Run(() =>
            {
               if (_session == null) {
                  _session = Session.LoadFromSavedModel(Path.Combine(_owner.ModelStorage, "saved_model"));
                  _labels = PbtxtParser.ParsePbtxtFile(Path.Combine(_owner.ModelStorage, "label_map.pbtxt"));
               }
               foreach (string img in data) {
                  var imgArr = ReadTensorFromImageFile(img);
                  var graph = _session.as_default().graph.as_default();
                  var t0 = graph.OperationByName("StatefulPartitionedCall").outputs;
                  Tensor tensorNum = t0[5];
                  Tensor tensorBoxes = t0[1];
                  Tensor tensorScores = t0[4];
                  Tensor tensorClasses = t0[2];
                  Tensor imgTensor = graph.OperationByName("serving_default_input_tensor").outputs[0];
                  Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };
                  var results = _session.run(outTensorArr, new FeedItem(imgTensor, imgArr));
                  BuildOutputImage(img, results);
               }
            }, cancellation);
            return null;
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
            var graph = _session.as_default().graph.as_default();
            var t0 = graph.OperationByName("StatefulPartitionedCall").outputs;
            var tensorNum = t0[5];
            var tensorBoxes = t0[1];
            var tensorScores = t0[4];
            var tensorClasses = t0[2];
            var imgTensor = graph.OperationByName("serving_default_input_tensor").outputs[0];
            var outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };
            // Dati di input e di output
            var input = data.ToDataViewGrid();
            var output = DataViewGrid.Create(this, OutputSchema);
            // Loop su tutte le righe di input
            foreach (var row in data.GetRowCursor(data.Schema).AsEnumerable()) {
               // Path dell'immagine
               var imagePath = row.ToDataViewValuesRow(this)["ImagePath"];
               // Trasforma l'immagine in dati numerici
               var imgArr = ReadTensorFromImageFile(imagePath);
               // La passa per la rete neurale
               cancellation.ThrowIfCancellationRequested();
               var results = _session.as_default().run(outTensorArr, new FeedItem(imgTensor, imgArr));
               // Ottiene i risultati
               cancellation.ThrowIfCancellationRequested();
               var scores = results[2].AsIterator<float>();
               var boxes = results[1].GetData<float>();
               var ids = np.squeeze(results[3]).GetData<float>();
               // Riempe la vista di dati di output con le rilevazioni
               for (var i = 0; i < scores.size; i++) {
                  var score = scores.MoveNext();
                  if (score < MIN_SCORE)
                     continue;
                  var label = _labels.Items.Where(w => w.Id == ids[i]).FirstOrDefault();
                  if (label == default)
                     continue;
                  output.Add(
                     ("ImagePath", imagePath),
                     ("ObjectId", label.Id),
                     ("ObjectName", label.Name),
                     ("ObjectDisplayName", label.DisplayName),
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
            _session = Session.LoadFromSavedModel(Path.Combine(_owner.ModelStorage, "saved_model"));
            _labels = PbtxtParser.ParsePbtxtFile(Path.Combine(_owner.ModelStorage, "label_map.pbtxt"));
            schema = OutputSchema;
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
         #region struct Rect
         /// <summary>
         /// Box contenitivo
         /// </summary>
         public struct Box
         {
            #region Fields
            /// <summary>
            /// Lato inferiore
            /// </summary>
            public float Bottom;
            /// <summary>
            /// Lato sinistro
            /// </summary>
            public float Left;
            /// <summary>
            /// Lato destro
            /// </summary>
            public float Right;
            /// <summary>
            /// Lato superiore
            /// </summary>
            public float Top;
            #endregion
         }
         #endregion
         #region Properties
         /// <summary>
         /// Tipi degli oggetti previsti
         /// </summary>
         public string[] Kind { get; }
         /// <summary>
         /// Punteggio per il tipo previsto
         /// </summary>
         public float[] Score { get; }
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
            //var dvg = DataViewGrid.Create(DataViewSchemaBuilder.Build
            //var grid = data.ToDataViewGrid();
            //Kind = grid[0]["PredictedLabel"];
            //var scores = (float[])grid[0]["Score"];
            //var slotNames = grid.Schema["Score"].GetSlotNames();
            //Scores = slotNames.Zip(scores).Select(item => new KeyValuePair<string, float>(item.First, item.Second)).ToArray();
            //Score = Scores.FirstOrDefault(s => s.Key == Kind).Value;
         }
         #endregion
      }
   }
}
