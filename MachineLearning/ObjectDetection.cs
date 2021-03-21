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
      public sealed class Model
      {
         #region Fields
         /// <summary>
         /// Contesto di machine learning
         /// </summary>
         private readonly IContextProvider<TFContext> _context;
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
         public float MIN_SCORE = 0.5f;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="context">Contesto di machine learning</param>
         internal Model(ObjectDetection owner, IContextProvider<TFContext> context)
         {
            _owner = owner;
            _context = context;
         }
         /// <summary>
         /// Crea l'immagine di output
         /// </summary>
         /// <param name="inputImagePath">Path dell'immagine di input</param>
         /// <param name="resultArr">Risultato della previsione</param>
         private void BuildOutputImage(string inputImagePath, NDArray[] resultArr)
         {
            // Elenco di oggetti del modello
            var pbTxtItems = PbtxtParser.ParsePbtxtFile(Path.Combine(_owner.ModelStorage, "label_map.pbtxt"));
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
                  var name = pbTxtItems.Items.Where(w => w.Id == id[i])
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
         public async Task<IDataAccess> GetPredictionDataAsync(IEnumerable<object> data, CancellationToken cancellation)
         {
            // Importa il grafico del modello
            await Task.Run(() =>
            {
               if (_session == null)
                  _session = Session.LoadFromSavedModel(Path.Combine(_owner.ModelStorage, "saved_model"));
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
         /// Legge un immagine e ne crea il tensore
         /// </summary>
         /// <param name="filePath">Path del file</param>
         /// <returns>Il tensore</returns>
         private NDArray ReadTensorFromImageFile(string filePath)
         {
            var tf = _context.Context;
            using var graph = tf.Graph().as_default();
            tf.compat.v1.disable_eager_execution();
            using var file_reader = tf.io.read_file(filePath, "file_reader");
            using var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            using var casted = tf.cast(decodeJpeg, TF_DataType.TF_UINT8);
            using var dims_expander = tf.expand_dims(casted, 0);
            tf.enable_eager_execution();
            using var sess = tf.Session(graph).as_default();
            return sess.run(dims_expander);
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
