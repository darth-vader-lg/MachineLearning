using MachineLearning;
using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MachineLearningStudio
{
   /// <summary>
   /// Pagina di test algoritmo Sdca per la previsione dei piedi
   /// </summary>
   public partial class PageFeetRegression : UserControl
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
      /// Previsore di significato testi
      /// </summary>
      private SizeRecognizer predictor;
      /// <summary>
      /// Task di previsione
      /// </summary>
      private (Task task, CancellationTokenSource cancellation) taskPrediction = (Task.CompletedTask, new CancellationTokenSource());
      /// <summary>
      /// Colore di background dei testi
      /// </summary>
      private readonly Color textBoxBackColor;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PageFeetRegression()
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
            // Forza il training
            MakePrediction(default, true);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Log del machine learning
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void Log(object sender, LoggingEventArgs e)
      {
         try {
            if (e.Kind < ChannelMessageKind.Info || e.Source != predictor.Name)
               return;
            predictor.Post(() =>
            {
               var (resel, SelectionStart, SelectionLength) = (textBoxOutput.SelectionStart < textBoxOutput.TextLength, textBoxOutput.SelectionStart, textBoxOutput.SelectionLength);
               var currentSelection = textBoxOutput.SelectionStart >= textBoxOutput.TextLength ? -1 : textBoxOutput.SelectionStart;
               textBoxOutput.AppendText(e.Message + Environment.NewLine);
               if (resel) {
                  textBoxOutput.Select(SelectionStart, SelectionLength);
                  textBoxOutput.ScrollToCaret();
               }
            });
         }
         catch (Exception) { }
      }
      /// <summary>
      /// Effettua la previsione in base ai dati impostati
      /// </summary>
      /// <param name="delay"></param>
      /// <param name="forceRebuildModel">Forza la ricostruzione del modello da zero</param>
      private void MakePrediction(TimeSpan delay = default, bool forceRebuildModel = false)
      {
         try {
            // Verifica che il controllo sia inizializzato
            if (!initialized)
               return;
            // Avvia un nuovo task di previsione
            taskPrediction.cancellation.Cancel();
            taskPrediction.cancellation = new CancellationTokenSource();
            taskPrediction.task = TaskPrediction(
               textBoxDataSetName.Text.Trim(),
               float.TryParse(textBoxLength.Text.Trim(), out var length) ? length : float.NaN,
               float.TryParse(textBoxInstep.Text.Trim(), out var instep) ? instep : float.NaN,
               taskPrediction.cancellation.Token,
               delay,
               forceRebuildModel);
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
            // Imposta le caselle di input
            textBoxDataSetName.Text = Settings.Default.PageFeetRegression.DataSetName?.Trim();
            textBoxInstep.Text = Settings.Default.PageFeetRegression.Instep.Trim();
            textBoxLength.Text = Settings.Default.PageFeetRegression.Length.Trim();
            // Crea il previsore
            predictor = new SizeRecognizer
            {
               AutoCommitData = true,
               AutoSaveModel = true,
               DataStorage = new DataStorageTextFile(Path.Combine(Environment.CurrentDirectory, "Data", textBoxDataSetName.Text)),
               ModelStorage = new ModelStorageFile(Path.Combine(Environment.CurrentDirectory, "Data", Path.ChangeExtension(textBoxDataSetName.Text, "model.zip"))),
               Name = "Predictor",
               TrainingData = new DataStorageTextMemory(),
            };
            predictor.SetInputSchema("Label", new TextLoader.Options
            {
               Columns = new[]
               {
                  new TextLoader.Column("Label", DataKind.Single, 0),
                  new TextLoader.Column("Length", DataKind.Single, 1),
                  new TextLoader.Column("Instep", DataKind.Single, 2),
               },
               Separators = new[] { ',' }
            });
            // Aggancia il log
            predictor.ML.NET.Log += Log;
            // Indicatore di inizializzazione ok
            initialized = true;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Task di previsione
      /// </summary>
      /// <param name="dataSetName">Nome del set di dati</param>
      /// <param name="length">Lunghezza aderente</param>
      /// <param name="instep">Calzata</param>
      /// <param name="cancel">Token di cancellazione</param>
      /// <param name="delay">Ritardo dell'avvio</param>
      /// <param name="forceRebuildModel">Forza la ricostruzione del modello da zero</param>
      /// <returns>Il task</returns>
      private async Task TaskPrediction(string dataSetName, float length, float instep, CancellationToken cancel, TimeSpan delay = default, bool forceRebuildModel = false)
      {
         try {
            // Attende la pausa
            await Task.Delay(delay, cancel);
            // Pulizia combo in caso di ricostruzione modello
            cancel.ThrowIfCancellationRequested();
            // Rilancia o avvia il task di training
            if (forceRebuildModel) {
               await predictor.StopTrainingAsync();
               _ = predictor.StartTrainingAsync(cancel);
            }
            cancel.ThrowIfCancellationRequested();
            if (float.IsNaN(length) || float.IsNaN(instep)) {
               labelNumberResult.Text = "";
               return;
            }
            labelNumberResult.Text = (await predictor.GetPredictionAsync(cancel, length, instep)).Size.ToString("0.#");
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            labelNumberResult.Text = "???";
            predictor.ML.NET.WriteLog(exc.ToString(), nameof(TaskPrediction));
         }
         //try { @@@
         //   cancel.ThrowIfCancellationRequested();
         //   var dataSetPath = dataSetName != null ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
         //   await Task.Run(() =>
         //   {
         //      try {
         //         cancel.ThrowIfCancellationRequested();
         //         // Verifica che non sia gia' stato calcolato il modello
         //         if (ml != null)
         //            return;
         //         // Crea il contesto
         //         ml = new MLContext(seed: 1);
         //         // Connette il log
         //         ml.Log += (sender, e) =>
         //         {
         //            try {
         //               if (e.Kind < ChannelMessageKind.Info)
         //                  return;
         //               textBoxOutput.BeginInvoke(new Action<LoggingEventArgs>(log =>
         //               {
         //                  try {
         //                     textBoxOutput.AppendText(log.Message + Environment.NewLine);
         //                     textBoxOutput.Select(textBoxOutput.TextLength, 0);
         //                     textBoxOutput.ScrollToCaret();
         //                  }
         //                  catch (Exception) { }
         //               }), e);
         //            }
         //            catch (Exception) { }
         //         };
         //         // Dati
         //         var dataView = default(IDataView);
         //         if (string.IsNullOrWhiteSpace(dataSetPath) || !File.Exists(dataSetPath)) {
         //            // Crea i dati
         //            var data = new List<PageFeetSdcaData>();
         //            var table = new[] {
         //               new { Number = 34.5f, LengthMin = 208, LengthMax = 211, InstepMin = 206, InstepMax = 223 },
         //               new { Number = 35f, LengthMin = 212, LengthMax = 215, InstepMin = 208, InstepMax = 225 },
         //               new { Number = 35.5f, LengthMin = 216, LengthMax = 220, InstepMin = 210, InstepMax = 227 },
         //               new { Number = 36f, LengthMin = 221, LengthMax = 225, InstepMin = 212, InstepMax = 229 },
         //               new { Number = 36.5f, LengthMin = 226, LengthMax = 229, InstepMin = 214, InstepMax = 231 },
         //               new { Number = 37f, LengthMin = 230, LengthMax = 233, InstepMin = 216, InstepMax = 233 },
         //               new { Number = 37.5f, LengthMin = 234, LengthMax = 237, InstepMin = 219, InstepMax = 236 },
         //               new { Number = 38f, LengthMin = 238, LengthMax = 241, InstepMin = 221, InstepMax = 238 },
         //               new { Number = 38.5f, LengthMin = 242, LengthMax = 245, InstepMin = 223, InstepMax = 240 },
         //               new { Number = 39f, LengthMin = 246, LengthMax = 249, InstepMin = 226, InstepMax = 243 },
         //               new { Number = 39.5f, LengthMin = 250, LengthMax = 254, InstepMin = 229, InstepMax = 246 },
         //               new { Number = 40f, LengthMin = 255, LengthMax = 258, InstepMin = 232, InstepMax = 249 },
         //               new { Number = 40.5f, LengthMin = 259, LengthMax = 262, InstepMin = 235, InstepMax = 252 },
         //               new { Number = 41f, LengthMin = 263, LengthMax = 267, InstepMin = 238, InstepMax = 255 },
         //               new { Number = 41.5f, LengthMin = 268, LengthMax = 271, InstepMin = 241, InstepMax = 258 },
         //               new { Number = 42f, LengthMin = 272, LengthMax = 275, InstepMin = 245, InstepMax = 262 },
         //               new { Number = 42.5f, LengthMin = 276, LengthMax = 280, InstepMin = 248, InstepMax = 265 },
         //               new { Number = 43f, LengthMin = 281, LengthMax = 284, InstepMin = 252, InstepMax = 269 },
         //               new { Number = 43.5f, LengthMin = 285, LengthMax = 288, InstepMin = 255, InstepMax = 272 }
         //            };
         //            foreach (var item in table) {
         //               for (var length = item.LengthMin; length <= item.LengthMax; length++) {
         //                  for (var instep = item.InstepMin; instep <= item.InstepMax; instep++)
         //                     data.Add(new PageFeetSdcaData { Number = item.Number, Instep = instep, Length = length });
         //               }
         //            }
         //            // Set di dati
         //            dataView = ml.Data.LoadFromEnumerable(data);
         //         }
         //         else {
         //            dataView = ml.Data.LoadFromTextFile<PageFeetSdcaData>(
         //               path: dataSetPath,
         //               hasHeader: false,
         //               separatorChar: ',',
         //               allowQuoting: true,
         //               allowSparse: false);
         //         }
         //         // Data process configuration with pipeline data transformations 
         //         cancel.ThrowIfCancellationRequested();
         //         var dataProcessPipeline =
         //            ml.Transforms.Categorical.OneHotHashEncoding(new[] { new InputOutputColumnPair(nameof(PageFeetSdcaData.Length), nameof(PageFeetSdcaData.Length)) }).
         //            Append(ml.Transforms.Concatenate("Features", new[] { nameof(PageFeetSdcaData.Length), nameof(PageFeetSdcaData.Instep) })).
         //            Append(ml.Transforms.NormalizeMinMax("Features", "Features")).
         //            AppendCacheCheckpoint(ml);
         //         // Set the training algorithm 
         //         cancel.ThrowIfCancellationRequested();
         //         var trainer = ml.Regression.Trainers.Sdca(
         //            new SdcaRegressionTrainer.Options() 
         //            { 
         //               L2Regularization = 1E-07f,
         //               L1Regularization = 0.25f,
         //               ConvergenceTolerance = 0.2f,
         //               Shuffle = true,
         //               BiasLearningRate = 1f,
         //               LabelColumnName = nameof(PageFeetSdcaData.Number),
         //               FeatureColumnName = "Features"
         //            });
         //         // Train the model
         //         var trainingPipeline = dataProcessPipeline.Append(trainer);
         //         cancel.ThrowIfCancellationRequested();
         //         var crossValidationResults = ml.Regression.CrossValidate( dataView, trainingPipeline, 5, nameof(PageFeetSdcaData.Number));
         //         ml.WriteLog(crossValidationResults.ToText(), "Cross validation average metrics");
         //         ml.WriteLog(crossValidationResults.Best().ToText(), "Best model metrics");
         //         model = crossValidationResults.Best().Model;
         //         // Salva il modello
         //         if (SaveModel) {
         //            cancel.ThrowIfCancellationRequested();
         //            ml.WriteLog($"Saving the model...", nameof(PageFeetSdca));
         //            ml.Model.Save(model, dataView.Schema, modelPath);
         //            ml.WriteLog($"The model is saved in {modelPath}", nameof(PageFeetSdca));
         //         }
         //         cancel.ThrowIfCancellationRequested();
         //         // Crea il generatore di previsioni
         //         predictor = ml.Model.CreatePredictionEngine<PageFeetSdcaData, PageFeetSdcaPrediction>(model);
         //      }
         //      catch (OperationCanceledException)
         //      {
         //         ml = null;
         //      }
         //      catch (Exception) {
         //         ml = null;
         //         throw;
         //      }
         //   }, cancel);
         //   cancel.ThrowIfCancellationRequested();
         //   if (!float.IsNaN(length) && !float.IsNaN(instep)) {
         //      // Aggiorna la previsione
         //      var prediction = predictor.Predict(new PageFeetSdcaData { Instep = instep, Length = length });
         //      labelNumberResult.Text = $"{prediction.Number:0.#}";
         //   }
         //   else
         //      labelNumberResult.Text = "";
         //}
         //catch (OperationCanceledException) { }
         //catch (Exception exc) {
         //   Trace.WriteLine(exc);
         //   labelNumberResult.Text = "???";
         //}
      }
      /// <summary>
      /// Evento di variazione del testo del nome del set di dati
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void textBoxDataSetName_TextChanged(object sender, EventArgs e)
      {
         try {
            if (sender is not TextBox tb)
               return;
            var path = Path.Combine(Environment.CurrentDirectory, "Data", tb.Text.Trim());
            if (!File.Exists(path)) {
               dataSetName = null;
               tb.BackColor = Color.Red;
            }
            else {
               tb.BackColor = textBoxBackColor;
               dataSetName = tb.Text.Trim();
               if (dataSetName != Settings.Default.PageFeetRegression.DataSetName) {
                  Settings.Default.PageFeetRegression.DataSetName = dataSetName;
                  Settings.Default.Save();
               }
            }
            MakePrediction(new TimeSpan(0, 0, 0, 0, 500), true);
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
            if (sender is not TextBox tb)
               return;
            labelNumberResult.Text = "";
            var text = tb.Text.Trim();
            if (!float.TryParse(text, out _))
               tb.BackColor = Color.Red;
            else {
               tb.BackColor = textBoxBackColor;
               if (text != Settings.Default.PageFeetRegression.Instep) {
                  Settings.Default.PageFeetRegression.Instep = text;
                  Settings.Default.Save();
                  MakePrediction(new TimeSpan(0, 0, 0, 0, 500));
               }
            }
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
            if (sender is not TextBox tb)
               return;
            labelNumberResult.Text = "";
            var text = tb.Text.Trim();
            if (!float.TryParse(text, out _))
               tb.BackColor = Color.Red;
            else {
               tb.BackColor = textBoxBackColor;
               if (text != Settings.Default.PageFeetRegression.Length) {
                  Settings.Default.PageFeetRegression.Length = text;
                  Settings.Default.Save();
                  MakePrediction(new TimeSpan(0, 0, 0, 0, 500));
               }
            }
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
      #region class PageFeetRegressionSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageFeetRegressionSettings
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
      public PageFeetRegressionSettings PageFeetRegression { get; set; } = new PageFeetRegressionSettings();
      #endregion
   }
}
