using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MachineLearningStudio
{
   /// <summary>
   /// Pagina di test algoritmo K-Means per la clusterizzazione dei piedi
   /// </summary>
   internal partial class PageFeetKMeans : UserControl
   {
      #region Fields
      /// <summary>
      /// Conversione da cluster a numeri di piede
      /// </summary>
      private string[] clusterToNumber;
      /// <summary>
      /// Set di dati
      /// </summary>
      private string dataSetName;
      /// <summary>
      /// Flag di controllo inizializzato
      /// </summary>
      private bool initialized;
      /// <summary>
      /// Calzata impostata
      /// </summary>
      private float instep = float.NaN;
      /// <summary>
      /// Lunghezza impostata
      /// </summary>
      private float length = float.NaN;
      /// <summary>
      /// Contesto ML
      /// </summary>
      private MLContext mlContext;
      /// <summary>
      /// Modello di apprendimento
      /// </summary>
      private ITransformer model;
      /// <summary>
      /// Path del modello di autoapprendimento dei piedi
      /// </summary>
      private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "FeedKMeans.zip");
      /// <summary>
      /// Punti per il plotting
      /// </summary>
      private Dictionary<uint, List<(double X, double Y)>> points;
      /// <summary>
      /// Previsore di piedi
      /// </summary>
      private PredictionEngine<PageFeetKMeansData, PageFeetKMeansPrediction> predictor;
      /// <summary>
      /// Task di previsione
      /// </summary>
      private (Task task, CancellationTokenSource cancellation) taskPrediction = (task: Task.CompletedTask, cancellation: new CancellationTokenSource());
      /// <summary>
      /// Colore di background dei testi
      /// </summary>
      private readonly Color textBoxBackColor;
      /// <summary>
      /// Limiti in x dei punti
      /// </summary>
      private (double min, double max) xLimits;
      /// <summary>
      /// Limiti in y dei punti
      /// </summary>
      private (double min, double max) yLimits;
      #endregion
      #region Properties
      /// <summary>
      /// Abilitazione al salvataggio del modello di training
      /// </summary>
      [Category("Behavior"), DefaultValue(false)]
      public bool SaveModel { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PageFeetKMeans()
      {
         InitializeComponent();
         textBoxBackColor = textBoxLength.BackColor;
      }
      /// <summary>
      /// Click sul pulsante di training
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void buttonTrain_Click(object sender, EventArgs e)
      {
         try {
            MakePrediction();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Effettua la previsione in base ai dati impostati
      /// </summary>
      private async void MakePrediction()
      {
         try {
            // Verifica che il controllo sia inizializzato
            if (!initialized)
               return;
            // Avvia un nuovo task di previsione
            taskPrediction.cancellation.Cancel();
            await taskPrediction.task;
            taskPrediction.cancellation = new CancellationTokenSource();
            taskPrediction.task = TaskPrediction(taskPrediction.cancellation.Token);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di caricamento del controllo
      /// </summary>
      /// <param name="e"></param>
      protected override void OnLoad(EventArgs e)
      {
         // Metodo base
         try {
            base.OnLoad(e);
            textBoxDataSetName.Text = Settings.Default.PageFeetKMeans.DataSetName?.Trim();
            textBoxInstep.Text = Settings.Default.PageFeetKMeans.Instep.Trim();
            textBoxLength.Text = Settings.Default.PageFeetKMeans.Length.Trim();
            initialized = true;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Evento di paint del pannello del grafico
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void panelPlot_Paint(object sender, PaintEventArgs e)
      {
         try {
            if (points == null)
               return;
            var colors = new[] { Color.Black, Color.Red, Color.Green, Color.Blue, Color.Yellow, Color.Lime, Color.Cyan };
            var brushes = colors.Select(c => new SolidBrush(c)).ToArray();
            var xFact = panelPlot.ClientSize.Width / (xLimits.max - xLimits.min);
            var yFact = panelPlot.ClientSize.Height / (yLimits.max - yLimits.min);
            foreach (var c in points) {
               foreach (var (X, Y) in c.Value)
                  e.Graphics.FillEllipse(brushes[(c.Key - 1) % brushes.Length], (float)((X - xLimits.min) * xFact), (float)((Y - yLimits.min) * yFact), 5f, 5f);
            }
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Evento di ridimensionamento del pannello
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void panelPlot_Resize(object sender, EventArgs e)
      {
         try {
            if (sender is Control control)
               control.Invalidate();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Task di previsione
      /// </summary>
      /// <param name="cancel"></param>
      /// <returns></returns>
      private async Task TaskPrediction(CancellationToken cancel)
      {
         try {
            cancel.ThrowIfCancellationRequested();
            var dataSetPath = dataSetName != null ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
            var plot = false;
            await Task.Run(() =>
            {
               try {
                  // Verifica che non sia gia' stato calcolato il modello
                  if (mlContext != null)
                     return;
                  // Crea il contesto
                  mlContext = new MLContext(seed: 0);
                  // Dizionario di contatori occorrenze numero / cluster
                  var idCounters = new Dictionary<string, long[]>();
                  // Dati
                  var dataView = default(IDataView);
                  if (string.IsNullOrWhiteSpace(dataSetPath) || !File.Exists(dataSetPath)) {
                     // Crea i dati
                     var data = new List<PageFeetKMeansData>();
                     var table = new[] {
                        new { Number = "34.5", LengthMin = 208, LengthMax = 211, InstepMin = 206, InstepMax = 223 },
                        new { Number = "35", LengthMin = 212, LengthMax = 215, InstepMin = 208, InstepMax = 225 },
                        new { Number = "35.5", LengthMin = 216, LengthMax = 220, InstepMin = 210, InstepMax = 227 },
                        new { Number = "36", LengthMin = 221, LengthMax = 225, InstepMin = 212, InstepMax = 229 },
                        new { Number = "36.5", LengthMin = 226, LengthMax = 229, InstepMin = 214, InstepMax = 231 },
                        new { Number = "37", LengthMin = 230, LengthMax = 233, InstepMin = 216, InstepMax = 233 },
                        new { Number = "37.5", LengthMin = 234, LengthMax = 237, InstepMin = 219, InstepMax = 236 },
                        new { Number = "38", LengthMin = 238, LengthMax = 241, InstepMin = 221, InstepMax = 238 },
                        new { Number = "38.5", LengthMin = 242, LengthMax = 245, InstepMin = 223, InstepMax = 240 },
                        new { Number = "39", LengthMin = 246, LengthMax = 249, InstepMin = 226, InstepMax = 243 },
                        new { Number = "39.5", LengthMin = 250, LengthMax = 254, InstepMin = 229, InstepMax = 246 },
                        new { Number = "40", LengthMin = 255, LengthMax = 258, InstepMin = 232, InstepMax = 249 },
                        new { Number = "40.5", LengthMin = 259, LengthMax = 262, InstepMin = 235, InstepMax = 252 },
                        new { Number = "41", LengthMin = 263, LengthMax = 267, InstepMin = 238, InstepMax = 255 },
                        new { Number = "41.5", LengthMin = 268, LengthMax = 271, InstepMin = 241, InstepMax = 258 },
                        new { Number = "42", LengthMin = 272, LengthMax = 275, InstepMin = 245, InstepMax = 262 },
                        new { Number = "42.5", LengthMin = 276, LengthMax = 280, InstepMin = 248, InstepMax = 265 },
                        new { Number = "43", LengthMin = 281, LengthMax = 284, InstepMin = 252, InstepMax = 269 },
                        new { Number = "43.5", LengthMin = 285, LengthMax = 288, InstepMin = 255, InstepMax = 272 }
                     };
                     foreach (var item in table) {
                        for (var length = item.LengthMin; length <= item.LengthMax; length++) {
                           for (var instep = item.InstepMin; instep <= item.InstepMax; instep++)
                              data.Add(new PageFeetKMeansData { Number = item.Number, Instep = instep, Length = length });
                        }
                     }
                     // Set di dati
                     dataView = mlContext.Data.LoadFromEnumerable(data);
                     //using (var stream = new FileStream(@"D:\Feet.txt", FileMode.Create))
                     //   mlContext.Data.SaveAsText(dataView, stream, ',', false, false);
                  }
                  else
                     dataView = mlContext.Data.LoadFromTextFile<PageFeetKMeansData>(dataSetPath, hasHeader: false, separatorChar: ',');
                  cancel.ThrowIfCancellationRequested();
                  // Numeri presenti nel set
                  var numbers = new HashSet<string>(from v in dataView.GetColumn<string>(nameof(PageFeetKMeansData.Number)) select v.Trim());
                  // Inizializzazione dizionario numeri / cluster
                  foreach (var n in numbers)
                     idCounters[n] = new long[numbers.Count];
                  // Elenco di associazioni cluster / numero
                  clusterToNumber = new string[numbers.Count];
                  // Crea colonna dati per il training
                  var dataCols = mlContext.Transforms.Concatenate("Features", nameof(PageFeetKMeansData.Length), nameof(PageFeetKMeansData.Instep));
                  // Crea il trainer di tipo KMeans
                  var trainer = mlContext.Clustering.Trainers.KMeans(new KMeansTrainer.Options()
                  {
                     AccelerationMemoryBudgetMb = 4096,
                     InitializationAlgorithm = KMeansTrainer.InitializationAlgorithm.KMeansYinyang,
                     NumberOfClusters = numbers.Count,
                     NumberOfThreads = 1,
                     FeatureColumnName = "Features",
                     OptimizationTolerance = 1E-7f,
                     MaximumNumberOfIterations = 10000,
                  });
                  // Crea la pipeline di training
                  var pipeline = dataCols.Append(trainer);
                  // Crea il modello di training
                  model = pipeline.Fit(dataView);
                  cancel.ThrowIfCancellationRequested();
                  // Salva il modello
                  if (SaveModel) {
                     using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                        mlContext.Model.Save(model, dataView.Schema, fileStream);
                  }
                  // Crea il previsore
                  predictor = mlContext.Model.CreatePredictionEngine<PageFeetKMeansData, PageFeetKMeansPrediction>(model);
                  cancel.ThrowIfCancellationRequested();
                  // Ottiene i centroidi
                  var centroids = default(VBuffer<float>[]);
                  ((TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>>)model).LastTransformer.Model.GetClusterCentroids(ref centroids, out int k);
                  // Estrae dal VBuffer pre semplicita'
                  var cleanCentroids = Enumerable.Range(1, clusterToNumber.Length).ToDictionary(x => (uint)x, x =>
                  {
                     var values = centroids[x - 1].GetValues().ToArray();
                     return values;
                  });
                  // Elenco di punti e limiti
                  points = new Dictionary<uint, List<(double X, double Y)>>();
                  xLimits = (double.MaxValue, -double.MaxValue);
                  yLimits = (double.MaxValue, -double.MaxValue);
                  // Cursore per spazzolare tutti i dati e creare le corrispondenze cluster / numero
                  var cursor = dataView.GetRowCursor(
                     new[] {
                        dataView.Schema[nameof(PageFeetKMeansData.Number)],
                        dataView.Schema[nameof(PageFeetKMeansData.Length)],
                        dataView.Schema[nameof(PageFeetKMeansData.Instep)]
                     });
                  // Spazzola tutti i dati effettuando le previsioni e incrementando i contatori relativi al cluster trovato per ciascun elemento
                  while (cursor.MoveNext()) {
                     // Legge i dati di riga
                     var number = default(ReadOnlyMemory<char>);
                     var length = 0f;
                     var instep = 0f;
                     cursor.GetGetter<ReadOnlyMemory<char>>(dataView.Schema[nameof(PageFeetKMeansData.Number)]).Invoke(ref number);
                     cursor.GetGetter<float>(dataView.Schema["Length"]).Invoke(ref length);
                     cursor.GetGetter<float>(dataView.Schema["Instep"]).Invoke(ref instep);
                     // Effettua la previsione utilizzando i dati di riga
                     var p = predictor.Predict(new PageFeetKMeansData { Number = number.ToString(), Length = length, Instep = instep });
                     // Incrementa il contatore di cluster relativo al numero specificato nella riga
                     idCounters[number.ToString()][p.PredictedClusterId - 1]++;
                     // Aggiunge il punto per il plotting
                     var weightedCentroid = cleanCentroids[p.PredictedClusterId].Zip(p.Distances, (x, y) => x * y);
                     var point = (X: weightedCentroid.Take(weightedCentroid.Count() / 2).Sum(), Y: weightedCentroid.Skip(weightedCentroid.Count() / 2).Sum());
                     if (!points.ContainsKey(p.PredictedClusterId))
                        points[p.PredictedClusterId] = new List<(double X, double Y)>();
                     points[p.PredictedClusterId].Add(point);
                     // Aggiorna i limiti di plotting
                     xLimits = (Math.Min(xLimits.min, point.X), Math.Max(xLimits.max, point.X));
                     yLimits = (Math.Min(yLimits.min, point.Y), Math.Max(yLimits.max, point.Y));
                     cancel.ThrowIfCancellationRequested();
                  }
                  // Crea l'associazione inversa iterando su ogni numero e scegliendo l'indice del cluster che contiene piu' occorrenze
                  foreach (var item in idCounters) {
                     var ix = 0;
                     clusterToNumber[item.Value.Select(val => new { count = val, ix = ix++ }).OrderByDescending(val => val.count).First().ix] = item.Key;
                  }
                  cancel.ThrowIfCancellationRequested();
                  plot = true;
               }
               catch (OperationCanceledException)
               {
                  mlContext = null;
               }
               catch (Exception) {
                  mlContext = null;
                  throw;
               }
            });
            // Invalida il pannello per il plotting
            if (plot)
               panelPlot.Invalidate();
            cancel.ThrowIfCancellationRequested();
            if (!float.IsNaN(length) && !float.IsNaN(instep)) {
               // Aggiorna la previsione
               var prediction = predictor.Predict(new PageFeetKMeansData { Instep = instep, Length = length });
               labelNumberResult.Text = clusterToNumber[prediction.PredictedClusterId - 1];
            }
            else
               labelNumberResult.Text = "";
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            labelNumberResult.Text = "???";
         }
      }
      /// <summary>
      /// Evento di variazione del testo del nome del set di dati
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void textBoxDataSetName_TextChanged(object sender, EventArgs e)
      {
         try {
            if (!(sender is TextBox tb))
               return;
            var path = Path.Combine(Environment.CurrentDirectory, "Data", tb.Text.Trim());
            if (!File.Exists(path)) {
               dataSetName = null;
               tb.BackColor = Color.Red;
            }
            else {
               tb.BackColor = textBoxBackColor;
               dataSetName = tb.Text.Trim();
               if (dataSetName != Settings.Default.PageFeetKMeans.DataSetName) {
                  Settings.Default.PageFeetKMeans.DataSetName = dataSetName;
                  Settings.Default.Save();
               }
            }
            mlContext = null;
            MakePrediction();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Evento di variazione del testo di lunghezza impostata
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void textBoxInstep_TextChanged(object sender, EventArgs e)
      {
         try {
            if (!(sender is TextBox tb))
               return;
            var text = tb.Text.Trim();
            if (!float.TryParse(text, out instep)) {
               instep = float.NaN;
               tb.BackColor = Color.Red;
            }
            else {
               tb.BackColor = textBoxBackColor;
               if (text != Settings.Default.PageFeetKMeans.Instep) {
                  Settings.Default.PageFeetKMeans.Instep = text;
                  Settings.Default.Save();
               }
            }
            MakePrediction();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Evento di variazione del testo di lunghezza impostata
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void textBoxLength_TextChanged(object sender, EventArgs e)
      {
         try {
            if (!(sender is TextBox tb))
               return;
            var text = tb.Text.Trim();
            if (!float.TryParse(text, out length)) {
               length = float.NaN;
               tb.BackColor = Color.Red;
            }
            else {
               tb.BackColor = textBoxBackColor;
               if (text != Settings.Default.PageFeetKMeans.Length) {
                  Settings.Default.PageFeetKMeans.Length = text;
                  Settings.Default.Save();
               }
            }
            MakePrediction();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }

   /// <summary>
   /// Impostazioni della pagina
   /// </summary>
   public partial class Settings
   {
      #region class PageFeetKMeansSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageFeetKMeansSettings
      {
         #region Properties
         /// <summary>
         /// Nome del set di dati
         /// </summary>
         public string DataSetName { get; set; } = "feet.data";
         /// <summary>
         /// Calzata
         /// </summary>
         public string Instep { get; set; } = "";
         /// <summary>
         /// Lunghezza
         /// </summary>
         public string Length { get; set; } = "";
         #endregion
      }
      #endregion
      #region Properties
      /// <summary>
      /// Settings della pagina
      /// </summary>
      public PageFeetKMeansSettings PageFeetKMeans { get; set; } = new PageFeetKMeansSettings();
      #endregion
   }
}
