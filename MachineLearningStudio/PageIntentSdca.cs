using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
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
   /// Pagina di test algoritmo Sdca per la previsione delle intenzioni
   /// </summary>
   public partial class PageIntentSdca : UserControl
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
      private ML ml;
      /// <summary>
      /// Modello di apprendimento
      /// </summary>
      private ITransformer model;
      /// <summary>
      /// Path del modello di autoapprendimento dei piedi
      /// </summary>
      private static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IntentSdca.zip");
      /// <summary>
      /// Previsore di piedi
      /// </summary>
      private PredictionEngine<PageIntentSdcaData, PageIntentSdcaPrediction> predictor;
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
      public PageIntentSdca()
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
            textBoxDataSetName.Text = Settings.Default.PageIntentSdca.DataSetName?.Trim();
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
            var sentence = textBoxSentence.Text;
            await Task.Run(() =>
            {
               try {
                  cancel.ThrowIfCancellationRequested();
                  // Verifica che non sia gia' stato calcolato il modello
                  if (ml != null)
                     return;
                  // Verifica che il set di dati sia valido
                  if (string.IsNullOrWhiteSpace(dataSetPath) || !File.Exists(dataSetPath))
                     return;
                  // Crea il contesto
                  ml = new ML(seed: 1);
                  // Connette il log
                  ml.LogMessage += (sender, e) =>
                  {
                     try {
                        if (e.Kind < ChannelMessageKind.Info)
                           return;
                        textBoxOutput.BeginInvoke(new Action<MLLogMessageEventArgs>(log =>
                        {
                           try {
                              textBoxOutput.AppendText(log.Text);
                              textBoxOutput.Select(textBoxOutput.TextLength, 0);
                              textBoxOutput.ScrollToCaret();
                           }
                           catch (Exception) { }
                        }), e);
                     }
                     catch (Exception) { }
                  };
                  // Dati
                  var dataView = ml.Context.Data.LoadFromTextFile<PageIntentSdcaData>(
                        path: dataSetPath,
                        hasHeader: false,
                        separatorChar: '|',
                        allowQuoting: true,
                        allowSparse: false);
                  // Data process configuration with pipeline data transformations 
                  cancel.ThrowIfCancellationRequested();
                  var dataProcessPipeline =
                     ml.Context.Transforms.Conversion.MapValueToKey(nameof(PageIntentSdcaData.Intent), nameof(PageIntentSdcaData.Intent)).
                     Append(ml.Context.Transforms.Text.FeaturizeText($"{nameof(PageIntentSdcaData.Sentence)}_tf", nameof(PageIntentSdcaData.Sentence))).
                     Append(ml.Context.Transforms.CopyColumns("Features", $"{nameof(PageIntentSdcaData.Sentence)}_tf")).
                     Append(ml.Context.Transforms.NormalizeMinMax("Features", "Features")).
                     AppendCacheCheckpoint(ml.Context);
                  // Set the training algorithm 
                  cancel.ThrowIfCancellationRequested();
                  var trainer = ml.Context.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                     new SdcaMaximumEntropyMulticlassTrainer.Options
                     {
                        L2Regularization = 1E-06f,
                        L1Regularization = 0.25f,
                        ConvergenceTolerance = 0.1f,
                        MaximumNumberOfIterations = 100,
                        Shuffle = false,
                        BiasLearningRate = 0f,
                        LabelColumnName = nameof(PageIntentSdcaData.Intent),
                        FeatureColumnName = "Features"
                     }).Append(ml.Context.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                  // Train the model
                  var trainingPipeline = dataProcessPipeline.Append(trainer);
                  cancel.ThrowIfCancellationRequested();
                  model = ml.EvaluateMulticlassClassification(dataView, trainingPipeline, nameof(PageIntentSdcaData.Intent));
                  // Salva il modello
                  if (SaveModel) {
                     cancel.ThrowIfCancellationRequested();
                     ml.SaveModel(dataView.Schema, model, modelPath);
                  }
                  cancel.ThrowIfCancellationRequested();
                  // Crea il generatore di previsioni
                  predictor = ml.Context.Model.CreatePredictionEngine<PageIntentSdcaData, PageIntentSdcaPrediction>(model);
               }
               catch (OperationCanceledException)
               {
                  ml = null;
                  predictor = null;
               }
               catch (Exception) {
                  ml = null;
                  predictor = null;
                  throw;
               }
            });
            cancel.ThrowIfCancellationRequested();
            if (predictor != null) {
               // Aggiorna la previsione
               var prediction = predictor.Predict(new PageIntentSdcaData { Sentence = sentence });
               labelIntentResult.Text = $"{prediction.Intent}";
            }
            else
               labelIntentResult.Text = "";
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            labelIntentResult.Text = "???";
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
               if (dataSetName != Settings.Default.PageIntentSdca.DataSetName) {
                  Settings.Default.PageIntentSdca.DataSetName = dataSetName;
                  Settings.Default.Save();
               }
            }
            ml = null;
            predictor = null;
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
      private void textBoxSentence_TextChanged(object sender, EventArgs e)
      {
         try {
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
      #region class PageIntentSdcaSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageIntentSdcaSettings
      {
         #region Properties
         /// <summary>
         /// Nome del set di dati
         /// </summary>
         public string DataSetName { get; set; } = "sentences.txt";
         #endregion
      }
      #endregion
      #region Properties
      /// <summary>
      /// Settings della pagina
      /// </summary>
      public PageIntentSdcaSettings PageIntentSdca { get; set; } = new PageIntentSdcaSettings();
      #endregion
   }
}
