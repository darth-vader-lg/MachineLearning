using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MachineLearningStudio
{
   /// <summary>
   /// Pagina di test algoritmo di classificazione immagini
   /// </summary>
   public partial class PageImageClassification : UserControl
   {
      #region Fields
      /// <summary>
      /// Abilitazione della validazione incrociata del modello
      /// </summary>
      private bool crossValidate;
      /// <summary>
      /// Directory delle immagini
      /// </summary>
      private string dataSetDir;
      /// <summary>
      /// Flag di controllo inizializzato
      /// </summary>
      private bool initialized;
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
      private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "ImageClassigication.zip");
      /// <summary>
      /// Previsore di piedi
      /// </summary>
      private PredictionEngine<PageImageClassificationData, PageImageClassificationPrediction> predictor;
      /// <summary>
      /// Task di previsione
      /// </summary>
      private (Task task, CancellationTokenSource cancellation) taskPrediction = (task: Task.CompletedTask, cancellation: new CancellationTokenSource());
      /// <summary>
      /// Colore di background dei testi
      /// </summary>
      private readonly Color textBoxBackColor;
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
      public PageImageClassification()
      {
         InitializeComponent();
         textBoxBackColor = textBoxImageSetName.BackColor;
      }
      /// <summary>
      /// Click sul pulsante di training
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void buttonLoad_Click(object sender, EventArgs e)
      {
         try {
            if (openFileDialog.ShowDialog(this) != DialogResult.OK)
               return;
            using (var fileData = new MemoryStream(File.ReadAllBytes(openFileDialog.FileName)))
               pictureBox.Image = Image.FromStream(fileData);
            MakePrediction();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Evento di modifica dello stato del check box di abilitazione della cross validation
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void checkBoxCrossValidate_CheckedChanged(object sender, EventArgs e)
      {
         try {
            if (!(sender is CheckBox cb))
               return;
            crossValidate = checkBoxCrossValidate.Checked;
            if (crossValidate != Settings.Default.PageImageClassification.CrossValidate) {
               Settings.Default.PageImageClassification.CrossValidate = crossValidate;
               Settings.Default.Save();
            }
            mlContext = null;
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
            textBoxImageSetName.Text = Settings.Default.PageImageClassification.DataSetDir?.Trim();
            initialized = true;
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
            var dataSetPath = dataSetDir != null ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetDir) : null;
            var enableCrossValidation = crossValidate;
            var ui = TaskScheduler.FromCurrentSynchronizationContext();
            await Task.Run(async () =>
            {
               try {
                  cancel.ThrowIfCancellationRequested();
                  // Verifica che non sia gia' stato calcolato il modello
                  if (mlContext != null)
                     return;
                  // Verifica che il path esista
                  if (!Directory.Exists(dataSetPath))
                     return;
                  // Crea il contesto
                  mlContext = new MLContext(seed: 1);
                  // Visualizzatore elaborazione
                  var sb = new StringBuilder();
                  // Dati
                  var dirs = from item in Directory.GetDirectories(dataSetPath, "*.*", SearchOption.TopDirectoryOnly)
                             where File.GetAttributes(item).HasFlag(FileAttributes.Directory)
                             select item;
                  var data = from dir in dirs
                             from file in Directory.GetFiles(dir, "*.*", SearchOption.TopDirectoryOnly)
                             let ext = Path.GetExtension(file).ToLower()
                             where new[] { ".jpg", ".png", ".bmp" }.Contains(ext)
                             select new PageImageClassificationData { Label = Path.GetFileName(dir), ImageSource = file };
                  cancel.ThrowIfCancellationRequested();
                  var dataView = mlContext.Data.LoadFromEnumerable(data);
                  // Data process configuration with pipeline data transformations 
                  cancel.ThrowIfCancellationRequested();
                  var dataProcessPipeline =
                     mlContext.Transforms.Conversion.MapValueToKey(nameof(PageImageClassificationData.Label), nameof(PageImageClassificationData.Label)).
                     Append(mlContext.Transforms.LoadRawImageBytes("ImageSource_featurized", null, nameof(PageImageClassificationData.ImageSource))).
                     Append(mlContext.Transforms.CopyColumns("Features", "ImageSource_featurized"));
                  // Set the training algorithm 
                  cancel.ThrowIfCancellationRequested();
                  var trainer =
                     mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: nameof(PageImageClassificationData.Label), featureColumnName: "Features").
                     Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                  var trainingPipeline = dataProcessPipeline.Append(trainer);
                  sb.AppendLine("=============== Training  model ===============");
                  var text = sb.ToString();
                  cancel.ThrowIfCancellationRequested();
                  await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  model = await Task.Run(() => trainingPipeline.Fit(dataView));
                  sb.AppendLine("=============== End of training process ===============");
                  text = sb.ToString();
                  cancel.ThrowIfCancellationRequested();
                  await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  // Salva il modello
                  if (SaveModel) {
                     cancel.ThrowIfCancellationRequested();
                     sb.AppendLine("================== Saving the model ===================");
                     text = sb.ToString();
                     cancel.ThrowIfCancellationRequested();
                     await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                     cancel.ThrowIfCancellationRequested();
                     using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                        mlContext.Model.Save(model, dataView.Schema, fileStream);
                     sb.AppendLine("==================== Model saved ======================");
                     text = sb.ToString();
                     cancel.ThrowIfCancellationRequested();
                     await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  }
                  if (enableCrossValidation) {
                     sb.AppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
                     text = sb.ToString();
                     cancel.ThrowIfCancellationRequested();
                     await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                     var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(dataView, trainingPipeline, numberOfFolds: 5, labelColumnName: nameof(PageImageClassificationData.Label));
                     var metricsInMultipleFolds = crossValidationResults.Select(r => r.Metrics);
                     var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
                     var microAccuracyAverage = microAccuracyValues.Average();
                     var sumOfSquaresOfDifferences = microAccuracyValues.Select(val => (val - microAccuracyAverage) * (val - microAccuracyAverage)).Sum();
                     var microAccuraciesStdDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (microAccuracyValues.Count() - 1));
                     var microAccuraciesConfidenceInterval95 = 1.96 * microAccuraciesStdDeviation / Math.Sqrt((microAccuracyValues.Count() - 1));
                     var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
                     var macroAccuracyAverage = macroAccuracyValues.Average();
                     sumOfSquaresOfDifferences = macroAccuracyValues.Select(val => (val - macroAccuracyAverage) * (val - macroAccuracyAverage)).Sum();
                     var macroAccuraciesStdDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (macroAccuracyValues.Count() - 1));
                     var macroAccuraciesConfidenceInterval95 = 1.96 * macroAccuraciesStdDeviation / Math.Sqrt((macroAccuracyValues.Count() - 1));
                     var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
                     var logLossAverage = logLossValues.Average();
                     sumOfSquaresOfDifferences = logLossValues.Select(val => (val - logLossAverage) * (val - logLossAverage)).Sum();
                     var logLossStdDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (logLossValues.Count() - 1));
                     var logLossConfidenceInterval95 = 1.96 * logLossStdDeviation / Math.Sqrt((logLossValues.Count() - 1));
                     var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
                     var logLossReductionAverage = logLossReductionValues.Average();
                     sumOfSquaresOfDifferences = logLossReductionValues.Select(val => (val - logLossReductionAverage) * (val - logLossReductionAverage)).Sum();
                     var logLossReductionStdDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (logLossReductionValues.Count() - 1));
                     var logLossReductionConfidenceInterval95 = 1.96 * logLossReductionStdDeviation / Math.Sqrt((logLossReductionValues.Count() - 1));
                     sb.AppendLine($"*************************************************************************************************************");
                     sb.AppendLine($"*       Metrics for Multi-class Classification model      ");
                     sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
                     sb.AppendLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
                     sb.AppendLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
                     sb.AppendLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
                     sb.AppendLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
                     sb.AppendLine($"*************************************************************************************************************");
                     text = sb.ToString();
                     cancel.ThrowIfCancellationRequested();
                     await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  }
                  cancel.ThrowIfCancellationRequested();
                  predictor = mlContext.Model.CreatePredictionEngine<PageImageClassificationData, PageImageClassificationPrediction>(model);
               }
               catch (OperationCanceledException)
               {
                  mlContext = null;
               }
               catch (Exception) {
                  mlContext = null;
                  throw;
               }
               await Task.CompletedTask;
            });
            cancel.ThrowIfCancellationRequested();
            if (!string.IsNullOrWhiteSpace(openFileDialog.FileName) && File.Exists(openFileDialog.FileName)) {
               var prediction = predictor.Predict(new PageImageClassificationData { ImageSource = openFileDialog.FileName });
               var labelBuffer = default(VBuffer<ReadOnlyMemory<char>>);
               predictor.OutputSchema["Score"].Annotations.GetValue("SlotNames", ref labelBuffer);
               var scoreIx = labelBuffer.Items().FirstOrDefault(item => item.Value.ToString() == prediction.Prediction).Key;
               labelClassResult.Text = $"{prediction.Prediction} ({(int)(prediction.Score[scoreIx] * 100f)}%)";
            }
            else
               labelClassResult.Text = "";
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            labelClassResult.Text = "???";
         }
      }
      /// <summary>
      /// Evento di variazione del testo del nome del set di dati
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void textBoxImageSetName_TextChanged(object sender, EventArgs e)
      {
         try {
            if (!(sender is TextBox tb))
               return;
            var path = Path.Combine(Environment.CurrentDirectory, "Data", tb.Text.Trim());
            if (!Directory.Exists(path)) {
               dataSetDir = null;
               tb.BackColor = Color.Red;
            }
            else {
               tb.BackColor = textBoxBackColor;
               dataSetDir = tb.Text.Trim();
               if (dataSetDir != Settings.Default.PageImageClassification.DataSetDir) {
                  Settings.Default.PageImageClassification.DataSetDir = dataSetDir;
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
      #endregion
   }

   /// <summary>
   /// Impostazioni della pagina
   /// </summary>
   public partial class Settings
   {
      #region class PageImageClassificationSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageImageClassificationSettings
      {
         #region Properties
         /// <summary>
         /// Abilitazione alla validazione incrociata del modello
         /// </summary>
         public bool CrossValidate { get; set; } = false;
         /// <summary>
         /// Nome del set di dati
         /// </summary>
         public string DataSetDir { get; set; } = "Images";
         #endregion
      }
      #endregion
      #region Properties
      /// <summary>
      /// Settings della pagina
      /// </summary>
      public PageImageClassificationSettings PageImageClassification { get; set; } = new PageImageClassificationSettings();
      #endregion
   }
}
