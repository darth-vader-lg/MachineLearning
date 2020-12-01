using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace MachineLearningStudio
{
   /// <summary>
   /// Pagina di test algoritmo K-Means per la clusterizzazione dei piedi
   /// </summary>
   public partial class PageIrisKMeans : UserControl
   {
      #region Fields
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
      private PredictionEngine<PageIrisKMeansData, PageIrisKMeansPrediction> irisPredictor;
      /// <summary>
      /// Contesto ML
      /// </summary>
      private MLContext mlContext;
      /// <summary>
      /// Modello di apprendimento
      /// </summary>
      private ITransformer model;
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
      public PageIrisKMeans()
      {
         InitializeComponent();
      }
      /// <summary>
      /// Evento di click sul pulsante di training
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void buttonTrain_Click(object sender, EventArgs e)
      {
         try {
            // Annulla tutto se il precedente set di previsioni non riguardava i fiori
            mlContext = new MLContext(seed: 0);
            var dataView = mlContext.Data.LoadFromTextFile<PageIrisKMeansData>(irisDataPath, hasHeader: false, separatorChar: ',');
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
            if (SaveModel) {
               using (var fileStream = new FileStream(irisModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                  mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
            //
            irisPredictor = irisPredictor != null && !ModifierKeys.HasFlag(Keys.Shift) ? irisPredictor : mlContext.Model.CreatePredictionEngine<PageIrisKMeansData, PageIrisKMeansPrediction>(model);
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
      #endregion
   }
}
