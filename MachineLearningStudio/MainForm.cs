using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace TestChoice
{
   public partial class MainForm : Form
   {
      #region Fields
      /// <summary>
      /// Conversione da cluster a numeri di piede
      /// </summary>
      private string[] feetClusterToNumber;
      /// <summary>
      /// Path dei dati dei fiori
      /// </summary>
      private static readonly string feetDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "feet.data");
      /// <summary>
      /// Path del modello di autoapprendimento dei piedi
      /// </summary>
      private static readonly string feetModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "FeedClusteringModel.zip");
      /// <summary>
      /// Previsore di piedi
      /// </summary>
      private PredictionEngine<FeetData, FeetPrediction> feetPredictor;
      /// <summary>
      /// Path dei dati dei fiori
      /// </summary>
      private static readonly string irisDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
      /// <summary>
      /// Path del modello di autoapprendimento degli iris
      /// </summary>
      private static readonly string irisModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
      /// <summary>
      /// Previsore di fiori
      /// </summary>
      private PredictionEngine<IrisData, IrisPrediction> irisPredictor;
      /// <summary>
      /// Contesto ML
      /// </summary>
      private MLContext mlContext;
      /// <summary>
      /// Modello di apprendimento
      /// </summary>
      private ITransformer model;
      /// <summary>
      /// Azione di plotting
      /// </summary>
      private Action<PaintEventArgs> plot;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public MainForm()
      {
         InitializeComponent();
      }
      /// <summary>
      /// Click sul pulsante di abilitazione plotting
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      [SuppressMessage("Style", "IDE1006:Stili di denominazione")]
      private void buttonPlot_Click(object sender, EventArgs e)
      {

      }
      /// <summary>
      /// Click sul pulsante di training dei piedi
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      [SuppressMessage("Style", "IDE1006:Stili di denominazione")]
      private void buttonTrainFeet_Click(object sender, EventArgs e)
      {
         try {
            // Annulla tutto se il precedente set di previsioni non riguardava i piedi
            if (irisPredictor == null)
               ResetML();
            // Crea il contesto
            mlContext = new MLContext(seed: 0);
            // Dizionario di contatori occorrenze numero / cluster
            var idCounters = new Dictionary<string, long[]>();
            // Verifica se un modello e' gia' esistente
            if (File.Exists(feetModelPath + "Caz") && !ModifierKeys.HasFlag(Keys.Shift)) {
               model = mlContext.Model.Load(feetModelPath, out _);
               feetPredictor = feetPredictor != null && !ModifierKeys.HasFlag(Keys.Shift) ? feetPredictor : mlContext.Model.CreatePredictionEngine<FeetData, FeetPrediction>(model);
            }
            // Creazione modello
            else {
               // Set di dati
               var dataView = mlContext.Data.LoadFromTextFile<FeetData>(feetDataPath, hasHeader: false, separatorChar: ',');
               // Numeri presenti nel set
               var numbers = new HashSet<string>(from v in dataView.GetColumn<string>(nameof(FeetData.Number)) select v.Trim());
               // Inizializzazione dizionario numeri / cluster
               foreach (var n in numbers)
                  idCounters[n] = new long[6];
               // Elenco di associazioni cluster / numero
               feetClusterToNumber = new string[numbers.Count];
               // Crea colonna dati per il training
               var dataCols = mlContext.Transforms.Concatenate("Features", nameof(FeetData.Length), nameof(FeetData.Instep));
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
               // Salva il modello
               //using (var fileStream = new FileStream(feetModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
               //   mlContext.Model.Save(model, dataView.Schema, fileStream);
               // Crea il previsore
               feetPredictor = mlContext.Model.CreatePredictionEngine<FeetData, FeetPrediction>(model);
               // Ottiene i centroidi
               var centroids = default(VBuffer<float>[]);
               ((TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>>)model).LastTransformer.Model.GetClusterCentroids(ref centroids, out int k);
               // Estrae dal VBuffer pre semplicita'
               var cleanCentroids = Enumerable.Range(1, feetClusterToNumber.Length).ToDictionary(x => (uint)x, x =>
               {
                  var values = centroids[x - 1].GetValues().ToArray();
                  return values;
               });
               // Elenco di punti e limiti
               var points = new Dictionary<uint, List<(double X, double Y)>>();
               var xLim = (Min: double.MaxValue, Max: -double.MaxValue);
               var yLim = (Min: double.MaxValue, Max: -double.MaxValue);
               // Cursore per spazzolare tutti i dati e creare le corrispondenze cluster / numero
               var cursor = dataView.GetRowCursor(
                  new[]
                  {
                     dataView.Schema[nameof(FeetData.Number)],
                     dataView.Schema[nameof(FeetData.Length)], 
                     dataView.Schema[nameof(FeetData.Instep)]
                  });
               // Spazzola tutti i dati effettuando le previsioni e incrementando i contatori relativi al cluster trovato per ciascun elemento
               while (cursor.MoveNext()) {
                  // Legge i dati di riga
                  var number = default(ReadOnlyMemory<char>);
                  var length = 0f;
                  var instep = 0f;
                  cursor.GetGetter<ReadOnlyMemory<char>>(dataView.Schema[nameof(FeetData.Number)]).Invoke(ref number);
                  cursor.GetGetter<float>(dataView.Schema["Length"]).Invoke(ref length);
                  cursor.GetGetter<float>(dataView.Schema["Instep"]).Invoke(ref instep);
                  // Effettua la previsione utilizzando i dati di riga
                  var p = feetPredictor.Predict(new FeetData { Number = number.ToString(), Length = length, Instep = instep });
                  // Incrementa il contatore di cluster relativo al numero specificato nella riga
                  idCounters[number.ToString()][p.PredictedClusterId - 1]++;
                  // Aggiunge il punto per il plotting
                  var weightedCentroid = cleanCentroids[p.PredictedClusterId].Zip(p.Distances, (x, y) => x * y);
                  var point = (X: weightedCentroid.Take(weightedCentroid.Count() / 2).Sum(), Y: weightedCentroid.Skip(weightedCentroid.Count() / 2).Sum());
                  if (!points.ContainsKey(p.PredictedClusterId))
                     points[p.PredictedClusterId] = new List<(double X, double Y)>();
                  points[p.PredictedClusterId].Add(point);
                  // Aggiorna i limiti di plotting
                  xLim.Min = Math.Min(xLim.Min, point.X);
                  xLim.Max = Math.Max(xLim.Max, point.X);
                  yLim.Min = Math.Min(yLim.Min, point.Y);
                  yLim.Max = Math.Max(yLim.Max, point.Y);
               }
               // Crea l'associazione inversa iterando su ogni numero e scegliendo l'indice del cluster che contiene piu' occorrenze
               foreach (var item in idCounters) {
                  var ix = 0;
                  feetClusterToNumber[item.Value.Select(val => new { count = val, ix = ix++ }).OrderByDescending(val => val.count).First().ix] = item.Key;
               }
               // Azione di plot
               plot = new Action<PaintEventArgs>(pe =>
               {
                  var colors = new[] { Color.Black, Color.Red, Color.Green, Color.Blue, Color.Yellow, Color.Lime, Color.Cyan };
                  var brushes = colors.Select(c => new SolidBrush(c)).ToArray();
                  var xFact = panelPlot.ClientSize.Width / (xLim.Max - xLim.Min);
                  var yFact = panelPlot.ClientSize.Height / (yLim.Max - yLim.Min);
                  foreach (var c in points) {
                     foreach (var p in c.Value)
                        pe.Graphics.FillEllipse(brushes[(c.Key - 1) % brushes.Length], (float)((p.X - xLim.Min) * xFact), (float)((p.Y - yLim.Min) * yFact), 5f, 5f);
                  }
               });
            }
            // Previsione di test
            var prediction = feetPredictor.Predict(new FeetData { Number = "36.5", Length = 227f, Instep = 222f});
            labelPrediction.Text = $"Number: {feetClusterToNumber[prediction.PredictedClusterId - 1]}" + $" Distances: {string.Join(" ", prediction.Distances)}";
            panelPlot.Invalidate();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Click sul pulsante di training dei fiori
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      [SuppressMessage("Style", "IDE1006:Stili di denominazione")]
      private void buttonTrainIris_Click(object sender, EventArgs e)
      {
         try {
            // Annulla tutto se il precedente set di previsioni non riguardava i fiori
            if (irisPredictor == null)
               ResetML();
            mlContext = mlContext != null && !ModifierKeys.HasFlag(Keys.Shift) ? mlContext : new MLContext(seed: 0);
            if (File.Exists(irisModelPath) && !ModifierKeys.HasFlag(Keys.Shift))
               model = mlContext.Model.Load(irisModelPath, out _);
            else {
               var dataView = mlContext.Data.LoadFromTextFile<IrisData>(irisDataPath, hasHeader: false, separatorChar: ',');
               var featuresColumnName = "Features";
               var km = mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3);
               var pipeline =
                  mlContext.Transforms.
                  Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth").
                  Append(mlContext.Clustering.Trainers.KMeans(new KMeansTrainer.Options()
                  {
                     AccelerationMemoryBudgetMb = 4096,
                     InitializationAlgorithm = KMeansTrainer.InitializationAlgorithm.KMeansYinyang,
                     NumberOfClusters = 3,
                     NumberOfThreads = 4,
                     FeatureColumnName = featuresColumnName,
                     MaximumNumberOfIterations = 10000,
                  }));
               //var pipeline =
               //   mlContext.Transforms.
               //   Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth").
               //   Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
               model = pipeline.Fit(dataView);
               using (var fileStream = new FileStream(irisModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                  mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
            //
            irisPredictor = irisPredictor != null && !ModifierKeys.HasFlag(Keys.Shift) ? irisPredictor : mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);
            var cols = irisPredictor.OutputSchema.ToArray();
            var prediction = irisPredictor.Predict(TestIrisData.Setosa);
            var prediction2 = irisPredictor.Predict(TestIrisData.Versicolor);
            var prediction3 = irisPredictor.Predict(TestIrisData.Virginica);
            labelPrediction.Text = $"Cluster: {prediction.PredictedClusterId}" + $" Distances: {string.Join(" ", prediction.Distances)}";
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Evento di paint del grafico
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      [SuppressMessage("Style", "IDE1006:Stili di denominazione")]
      private void panelPlot_Paint(object sender, PaintEventArgs e)
      {
         try {
            // Chiama l'azione di plotting
            plot?.Invoke(e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Resize del pannello del plot
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      [SuppressMessage("Style", "IDE1006:Stili di denominazione")]
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
      /// Resetta il machine learning
      /// </summary>
      private void ResetML()
      {
         try {
            mlContext = null;
            feetPredictor = null;
            irisPredictor = null;
            plot = null;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      #endregion
   }
}
