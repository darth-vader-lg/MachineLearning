using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
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
   /// Pagina di test algoritmo Sdca per la previsione dei piedi
   /// </summary>
   public partial class PageFeetSdca : UserControl
   {
      #region Fields
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
      private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "FeetSdca.zip");
      /// <summary>
      /// Previsore di piedi
      /// </summary>
      private PredictionEngine<PageFeetSdcaData, PageFeetSdcaPrediction> predictor;
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
      public PageFeetSdca()
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
            textBoxDataSetName.Text = Settings.Default.PageFeetSdca.DataSetName?.Trim();
            textBoxInstep.Text = Settings.Default.PageFeetSdca.Instep.Trim();
            textBoxLength.Text = Settings.Default.PageFeetSdca.Length.Trim();
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
            var dataSetPath = dataSetName != null ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
            var ui = TaskScheduler.FromCurrentSynchronizationContext();
            await Task.Run(async () =>
            {
               try {
                  cancel.ThrowIfCancellationRequested();
                  // Verifica che non sia gia' stato calcolato il modello
                  if (mlContext != null)
                     return;
                  // Crea il contesto
                  mlContext = new MLContext(seed: 1);
                  // Visualizzatore elaborazione
                  var sb = new StringBuilder();
                  // Dati
                  var dataView = default(IDataView);
                  if (string.IsNullOrWhiteSpace(dataSetPath) || !File.Exists(dataSetPath)) {
                     // Crea i dati
                     var data = new List<PageFeetSdcaData>();
                     var table = new[] {
                        new { Number = 34.5f, LengthMin = 208, LengthMax = 211, InstepMin = 206, InstepMax = 223 },
                        new { Number = 35f, LengthMin = 212, LengthMax = 215, InstepMin = 208, InstepMax = 225 },
                        new { Number = 35.5f, LengthMin = 216, LengthMax = 220, InstepMin = 210, InstepMax = 227 },
                        new { Number = 36f, LengthMin = 221, LengthMax = 225, InstepMin = 212, InstepMax = 229 },
                        new { Number = 36.5f, LengthMin = 226, LengthMax = 229, InstepMin = 214, InstepMax = 231 },
                        new { Number = 37f, LengthMin = 230, LengthMax = 233, InstepMin = 216, InstepMax = 233 },
                        new { Number = 37.5f, LengthMin = 234, LengthMax = 237, InstepMin = 219, InstepMax = 236 },
                        new { Number = 38f, LengthMin = 238, LengthMax = 241, InstepMin = 221, InstepMax = 238 },
                        new { Number = 38.5f, LengthMin = 242, LengthMax = 245, InstepMin = 223, InstepMax = 240 },
                        new { Number = 39f, LengthMin = 246, LengthMax = 249, InstepMin = 226, InstepMax = 243 },
                        new { Number = 39.5f, LengthMin = 250, LengthMax = 254, InstepMin = 229, InstepMax = 246 },
                        new { Number = 40f, LengthMin = 255, LengthMax = 258, InstepMin = 232, InstepMax = 249 },
                        new { Number = 40.5f, LengthMin = 259, LengthMax = 262, InstepMin = 235, InstepMax = 252 },
                        new { Number = 41f, LengthMin = 263, LengthMax = 267, InstepMin = 238, InstepMax = 255 },
                        new { Number = 41.5f, LengthMin = 268, LengthMax = 271, InstepMin = 241, InstepMax = 258 },
                        new { Number = 42f, LengthMin = 272, LengthMax = 275, InstepMin = 245, InstepMax = 262 },
                        new { Number = 42.5f, LengthMin = 276, LengthMax = 280, InstepMin = 248, InstepMax = 265 },
                        new { Number = 43f, LengthMin = 281, LengthMax = 284, InstepMin = 252, InstepMax = 269 },
                        new { Number = 43.5f, LengthMin = 285, LengthMax = 288, InstepMin = 255, InstepMax = 272 }
                     };
                     foreach (var item in table) {
                        for (var length = item.LengthMin; length <= item.LengthMax; length++) {
                           for (var instep = item.InstepMin; instep <= item.InstepMax; instep++)
                              data.Add(new PageFeetSdcaData { Number = item.Number, Instep = instep, Length = length });
                        }
                     }
                     // Set di dati
                     dataView = mlContext.Data.LoadFromEnumerable(data);
                  }
                  else {
                     dataView = mlContext.Data.LoadFromTextFile<PageFeetSdcaData>(
                        path: dataSetPath,
                        hasHeader: false,
                        separatorChar: ',',
                        allowQuoting: true,
                        allowSparse: false);
                  }
                  // Data process configuration with pipeline data transformations 
                  cancel.ThrowIfCancellationRequested();
                  var dataProcessPipeline =
                     mlContext.Transforms.Categorical.OneHotHashEncoding(new[] { new InputOutputColumnPair(nameof(PageFeetSdcaData.Length), nameof(PageFeetSdcaData.Length)) }).
                     Append(mlContext.Transforms.Concatenate("Features", new[] { nameof(PageFeetSdcaData.Length), nameof(PageFeetSdcaData.Instep) })).
                     Append(mlContext.Transforms.NormalizeMinMax("Features", "Features")).
                     AppendCacheCheckpoint(mlContext);
                  // Set the training algorithm 
                  cancel.ThrowIfCancellationRequested();
                  var trainer = mlContext.Regression.Trainers.Sdca(
                     new SdcaRegressionTrainer.Options() 
                     { 
                        L2Regularization = 1E-07f,
                        L1Regularization = 0.25f,
                        ConvergenceTolerance = 0.2f,
                        Shuffle = true,
                        BiasLearningRate = 1f,
                        LabelColumnName = nameof(PageFeetSdcaData.Number),
                        FeatureColumnName = "Features"
                     });
                  var trainingPipeline = dataProcessPipeline.Append(trainer);
                  cancel.ThrowIfCancellationRequested();
                  sb.AppendLine("=============== Training  model ===============");
                  var text = sb.ToString();
                  await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  model = await Task.Run(() => trainingPipeline.Fit(dataView));
                  sb.AppendLine("=============== End of training process ===============");
                  text = sb.ToString();
                  await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  // Salva il modello
                  if (SaveModel) {
                     cancel.ThrowIfCancellationRequested();
                     sb.AppendLine("================== Saving the model ===================");
                     text = sb.ToString();
                     await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                     using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                        mlContext.Model.Save(model, dataView.Schema, fileStream);
                     sb.AppendLine("==================== Model saved ======================");
                     text = sb.ToString();
                     await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  }
                  sb.AppendLine("=============== Cross-validating to get model's accuracy metrics ===============");
                  text = sb.ToString();
                  await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  cancel.ThrowIfCancellationRequested();
                  var crossValidationResults = await Task.Run(() => mlContext.Regression.CrossValidate(dataView, trainingPipeline, numberOfFolds: 5, labelColumnName: nameof(PageFeetSdcaData.Number)));
                  var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
                  var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
                  var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
                  var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
                  var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);
                  sb.AppendLine($"*************************************************************************************************************");
                  sb.AppendLine($"*       Metrics for Regression model      ");
                  sb.AppendLine($"*------------------------------------------------------------------------------------------------------------");
                  sb.AppendLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
                  sb.AppendLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
                  sb.AppendLine($"*       Average RMS:           {RMS.Average():0.###}  ");
                  sb.AppendLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
                  sb.AppendLine($"*       Average R-squared:     {R2.Average():0.###}  ");
                  sb.AppendLine($"*************************************************************************************************************");
                  text = sb.ToString();
                  await Task.Factory.StartNew(() => textBoxOutput.Text = text, CancellationToken.None, TaskCreationOptions.None, ui);
                  cancel.ThrowIfCancellationRequested();
                  predictor = mlContext.Model.CreatePredictionEngine<PageFeetSdcaData, PageFeetSdcaPrediction>(model);
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
            cancel.ThrowIfCancellationRequested();
            if (!float.IsNaN(length) && !float.IsNaN(instep)) {
               // Aggiorna la previsione
               var prediction = predictor.Predict(new PageFeetSdcaData { Instep = instep, Length = length });
               labelNumberResult.Text = $"{prediction.Number}";
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
               if (dataSetName != Settings.Default.PageFeetSdca.DataSetName) {
                  Settings.Default.PageFeetSdca.DataSetName = dataSetName;
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
               if (text != Settings.Default.PageFeetSdca.Instep) {
                  Settings.Default.PageFeetSdca.Instep = text;
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
               if (text != Settings.Default.PageFeetSdca.Length) {
                  Settings.Default.PageFeetSdca.Length = text;
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
      #region class PageFeetSdcaSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageFeetSdcaSettings
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
      public PageFeetSdcaSettings PageFeetSdca { get; set; } = new PageFeetSdcaSettings();
      #endregion
   }
}
