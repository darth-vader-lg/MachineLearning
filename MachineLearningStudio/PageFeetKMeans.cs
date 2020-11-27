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
using System.Windows.Forms;

namespace TestChoice
{
   /// <summary>
   /// Pagina di test algoritmo K-Means per la clusterizzazione dei piedi
   /// </summary>
   public partial class PageFeetKMeans : UserControl
   {
      #region Fields
      /// <summary>
      /// Conversione da cluster a numeri di piede
      /// </summary>
      private string[] clusterToNumber;
      /// <summary>
      /// Path dei dati dei fiori
      /// </summary>
      private static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "feet.data");
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
      private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "FeedClusteringModel.zip");
      /// <summary>
      /// Punti per il plotting
      /// </summary>
      private Dictionary<uint, List<(double X, double Y)>> points;
      /// <summary>
      /// Previsore di piedi
      /// </summary>
      private PredictionEngine<FeetData, FeetPrediction> predictor;
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
      }
      /// <summary>
      /// Click sul pulsante di training
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void buttonTrain_Click(object sender, EventArgs e)
      {
         try {
            // Crea il contesto
            mlContext = new MLContext(seed: 0);
            // Dizionario di contatori occorrenze numero / cluster
            var idCounters = new Dictionary<string, long[]>();
            // Set di dati
            var dataView = mlContext.Data.LoadFromTextFile<FeetData>(dataPath, hasHeader: false, separatorChar: ',');
            // Numeri presenti nel set
            var numbers = new HashSet<string>(from v in dataView.GetColumn<string>(nameof(FeetData.Number)) select v.Trim());
            // Inizializzazione dizionario numeri / cluster
            foreach (var n in numbers)
               idCounters[n] = new long[6];
            // Elenco di associazioni cluster / numero
            clusterToNumber = new string[numbers.Count];
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
            if (SaveModel) {
               using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                  mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
            // Crea il previsore
            predictor = mlContext.Model.CreatePredictionEngine<FeetData, FeetPrediction>(model);
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
               var p = predictor.Predict(new FeetData { Number = number.ToString(), Length = length, Instep = instep });
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
            }
            // Crea l'associazione inversa iterando su ogni numero e scegliendo l'indice del cluster che contiene piu' occorrenze
            foreach (var item in idCounters) {
               var ix = 0;
               clusterToNumber[item.Value.Select(val => new { count = val, ix = ix++ }).OrderByDescending(val => val.count).First().ix] = item.Key;
            }
            // Invalida il pannello per il plotting
            panelPlot.Invalidate();
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
      #endregion
   }
}
