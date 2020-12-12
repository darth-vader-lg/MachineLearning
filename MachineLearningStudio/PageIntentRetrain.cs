using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using ML.Utilities;
using System;
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
   /// Pagina di test algoritmo per la previsione delle intenzioni
   /// </summary>
   public partial class PageIntentRetrain : UserControl
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
      /// Contesto ML
      /// </summary>
      private MLContext ml;
      /// <summary>
      /// Modello di apprendimento
      /// </summary>
      private (ITransformer In, ITransformer Trained, ITransformer Out) model = default;
      /// <summary>
      /// Task di previsione
      /// </summary>
      private (Task task, CancellationTokenSource cancellation) taskPrediction = (task: Task.CompletedTask, cancellation: new CancellationTokenSource());
      /// <summary>
      /// Colore di background dei testi
      /// </summary>
      private readonly Color textBoxBackColor;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PageIntentRetrain()
      {
         InitializeComponent();
         textBoxBackColor = textBoxSentence.BackColor;
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
            ml = null;
            MakePrediction(default, true);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
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
            taskPrediction.task = TaskPrediction(taskPrediction.cancellation.Token, delay, forceRebuildModel);
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
            textBoxDataSetName.Text = Settings.Default.PageIntent.DataSetName?.Trim();
            initialized = true;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Task di previsione
      /// </summary>
      /// <param name="cancel">Token di cancellazione</param>
      /// <param name="delay">Ritardo dell'avvio</param>
      /// <param name="forceRebuildModel">Forza la ricostruzione del modello da zero</param>
      /// <returns>Il task</returns>
      private async Task TaskPrediction(CancellationToken cancel, TimeSpan delay = default, bool forceRebuildModel = false)
      {
         try {
            await Task.Delay(delay, cancel);
            // Pulizia combo in caso di ricostruzione modello
            cancel.ThrowIfCancellationRequested();
            ml = forceRebuildModel ? null : ml;
            if (ml == null)
               textBoxOutput.Clear();
            // Path del set di dati in chiaro
            var dataSetPath = !string.IsNullOrWhiteSpace(dataSetName) ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
            // Path del modello
            var modelInPath = !string.IsNullOrWhiteSpace(dataSetPath) ? Path.ChangeExtension(dataSetPath, "model-in.zip") : null;
            var modelTrainedPath = !string.IsNullOrWhiteSpace(dataSetPath) ? Path.ChangeExtension(dataSetPath, "model-trained.zip") : null;
            var modelOutPath = !string.IsNullOrWhiteSpace(dataSetPath) ? Path.ChangeExtension(dataSetPath, "model-out.zip") : null;
            // Sentenza attuale
            var sentence = textBoxSentence.Text;
            // Task di costruzione del modello
            await Task.Run(() =>
            {
               try {
                  cancel.ThrowIfCancellationRequested();
                  // Verifica che non sia gia' stato calcolato il modello
                  if (ml != null || (dataSetPath == null && (modelInPath == null || modelTrainedPath == null || modelOutPath == null)))
                     return;
                  // Abilitazione al caricamento del modello preesistente
                  var loadModel = !forceRebuildModel && new[] { modelInPath, modelTrainedPath, modelOutPath }.All(item => File.Exists(item) && (!File.Exists(dataSetPath) || File.GetLastWriteTime(dataSetPath) < File.GetLastWriteTime(item)));
                  // Abilitazione al caricamento di dati
                  var loadData = File.Exists(dataSetPath);
                  // Verifica le condizioni di caricamento possibile
                  if (!loadModel && !loadData)
                     return;
                  // Crea il contesto
                  ml = new MLContext(seed: 1);
                  // Connette il log
                  var logSourceFilter = default(string[]);
                  ml.Log += (sender, e) =>
                  {
                     try {
                        if (e.Kind < ChannelMessageKind.Info)
                           return;
                        if (logSourceFilter != null) {
                           if (logSourceFilter.FirstOrDefault(item => item == e.Source) == null)
                              return;
                        }
                        textBoxOutput.Invoke(new Action<LoggingEventArgs>(log =>
                        {
                           try {
                              var scroll = textBoxOutput.SelectionStart >= textBoxOutput.TextLength;
                              textBoxOutput.AppendText(log.Message + Environment.NewLine);
                              if (scroll) {
                                 textBoxOutput.Select(textBoxOutput.TextLength, 0);
                                 textBoxOutput.ScrollToCaret();
                              }
                           }
                           catch (Exception) { }
                        }), e);
                     }
                     catch (Exception) { }
                  };
                  // Verifica se deve caricare un modello preesistente
                  if (loadModel) {
                     try {
                        // Carica il modello
                        ml.WriteLog($"Loading the model from {modelInPath}...", GetType().Name);
                        model.In = ml.Model.Load(modelInPath, out var _);
                        ml.WriteLog("The model is loaded", GetType().Name);
                        ml.WriteLog($"Loading the model from {modelTrainedPath}...", GetType().Name);
                        model.Trained = ml.Model.Load(modelTrainedPath, out var _);
                        ml.WriteLog("The model is loaded", GetType().Name);
                        ml.WriteLog($"Loading the model from {modelOutPath}...", GetType().Name);
                        model.Out = ml.Model.Load(modelOutPath, out var _);
                        ml.WriteLog("The model is loaded", GetType().Name);
                        // Disabilita il caricamento dei dati in chiaro
                        loadData = false;
                     }
                     catch (Exception) {  }
                  }
                  // Verifica se deve caricare dai dati
                  if (loadData) {
                     // Crea l'input di dati
                     var dataView = ml.Data.LoadFromTextFile(
                        path: dataSetPath,
                        columns: new[]
                        {
                           new TextLoader.Column("Intent", DataKind.String, 0),
                           new TextLoader.Column("Sentence", DataKind.String, 1),
                        },
                        hasHeader: false,
                        separatorChar: '|',
                        allowQuoting: true,
                        allowSparse: false);
                     // Pipeline di trasformazione dei dati
                     cancel.ThrowIfCancellationRequested();
                     var modelInPipeline =
                        ml.Transforms.Conversion.MapValueToKey("Intent").
                        Append(ml.Transforms.Text.FeaturizeText("Sentence_tf", "Sentence")).
                        Append(ml.Transforms.CopyColumns("Features", "Sentence_tf")).
                        Append(ml.Transforms.NormalizeMinMax("Features")).
                        AppendCacheCheckpoint(ml);
                     model.In = modelInPipeline.Fit(dataView);
                     // Algoritmo di training
                     cancel.ThrowIfCancellationRequested();
                     var trainerOptions = new LbfgsMaximumEntropyMulticlassTrainer.Options
                     {
                        HistorySize = 1000,
                        OptimizationTolerance = 1E-06f,
                        ShowTrainingStatistics = true,
                        L2Regularization = 1E-06f,
                        L1Regularization = 0.25f,
                        MaximumNumberOfIterations = 100,
                        LabelColumnName = nameof(PageIntentData.Intent),
                        FeatureColumnName = "Features",
                        NumberOfThreads = Environment.ProcessorCount,
                     };
                     var trainer = ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy(trainerOptions);
                     // Pipeline di training
                     var dataIn = model.In.Transform(dataView);
                     var trainingPipeline = ml.Transforms.SelectColumns((from col in dataIn.Schema select col.Name).ToArray()).Append(trainer);
                     // Effettua la miglior valutazione del modello
                     cancel.ThrowIfCancellationRequested();
                     var crossValidationResults = ml.MulticlassClassification.CrossValidate(dataIn, trainingPipeline, 50, "Intent");
                     ml.WriteLog(crossValidationResults.ToText(), "Cross validation average metrics");
                     ml.WriteLog(crossValidationResults.Best().ToText(), "Best model metrics");
                     model.Trained = crossValidationResults.Best().Model;
                     var dataTrained = model.Trained.Transform(dataIn);
                     var modelOutPipeline = ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel");
                     model.Out = modelOutPipeline.Fit(dataTrained);
                     // Salva il modello
                     cancel.ThrowIfCancellationRequested();
                     ml.WriteLog($"Saving the model to {modelInPath}...", GetType().Name);
                     ml.Model.Save(model.In, dataView.Schema, modelInPath);
                     ml.WriteLog("The model is saved", GetType().Name);
                     ml.WriteLog($"Saving the model to {modelTrainedPath}...", GetType().Name);
                     ml.Model.Save(model.Trained, dataIn.Schema, modelTrainedPath);
                     ml.WriteLog("The model is saved", GetType().Name);
                     ml.WriteLog($"Saving the model to {modelOutPath}...", GetType().Name);
                     ml.Model.Save(model.Out, dataTrained.Schema, modelOutPath);
                     ml.WriteLog("The model is saved", GetType().Name);
                  }
               }
               catch (OperationCanceledException)
               {
                  ml = null;
               }
               catch (Exception exc) {
                  try {
                     ml.WriteLog(exc.ToString(), GetType().Name);
                  }
                  catch (Exception) { }
                  ml = null;
                  throw;
               }
            }, cancel);
            // Verifica se esiste un gestore di previsioni
            cancel.ThrowIfCancellationRequested();
            if (model.In != null && model.Trained != null && model.Out != null && !string.IsNullOrWhiteSpace(sentence)) {
               var previewData = ml.Data.LoadFromEnumerable(new[]
               {
                  new { Intent = "?", Sentence = sentence }
               });
               var previewDataIn = model.In.Transform(previewData);
               var previewDataTrained = model.Trained.Transform(previewDataIn);
               var previewDataOut = model.Out.Transform(previewDataTrained);
               textBoxIntent.Text = previewDataOut.GetString("PredictedLabel");
               textBoxIntent.BackColor = textBoxBackColor;
            }
            else {
               textBoxIntent.Text = "";
               textBoxIntent.BackColor = Color.Red;
            }
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            textBoxIntent.Text = "";
            textBoxIntent.BackColor = Color.Red;
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
            // Combo box generatore evento
            if (sender is not TextBox tb)
               return;
            // Path del set di dati
            var path = Path.Combine(Environment.CurrentDirectory, "Data", tb.Text.Trim());
            // Verifica se file non esistente
            if (!File.Exists(path)) {
               // Visualizza l'errore
               dataSetName = null;
               tb.BackColor = Color.Red;
            }
            else {
               // Aggiorna il set di dati
               tb.BackColor = textBoxBackColor;
               dataSetName = tb.Text.Trim();
               // Salva nei settings lo stato
               if (dataSetName != Settings.Default.PageIntent.DataSetName) {
                  Settings.Default.PageIntent.DataSetName = dataSetName;
                  Settings.Default.Save();
               }
            }
            // Avvia una ricostruzione del modello
            ml = null;
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
      private void textBoxSentence_TextChanged(object sender, EventArgs e)
      {
         try {
            // Lancia una previsione
            MakePrediction(new TimeSpan(0, 0, 0, 0, 500));
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
      #region class PageIntentRetrainSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageIntentRetrainSettings
      {
         #region Properties
         /// <summary>
         /// Nome del set di dati
         /// </summary>
         public string DataSetName { get; set; } = "intents.data";
         #endregion
      }
      #endregion
      #region Properties
      /// <summary>
      /// Settings della pagina
      /// </summary>
      public PageIntentRetrainSettings PageIntentRetrain { get; set; } = new PageIntentRetrainSettings();
      #endregion
   }
}
