using Microsoft.ML;
using Microsoft.ML.Data;
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
      /// Evento di termine selezione item di una combo box
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void comboBoxIntent_SelectionChangeCommitted(object sender, EventArgs e)
      {
         try {
            // Verifiche iniziali
            if (dataSetName == null || !(sender is ComboBox cb))
               return;
            // Path del set di dati
            var dataSetPath = dataSetName != null ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
            // Verifica esistenza
            if (!File.Exists(dataSetPath))
               return;
            // Contenuto del set di dati
            var dataSetBytes = File.ReadAllBytes(dataSetPath);
            // Directory inizialie di ricerca a ritroso del path della sorgente dei dati
            var dataSourceDir = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..\\"));
            var dataSourcePath = default(string);
            // Ricerca verso la radice il file originale dei dati per la sincronizzazione (solo se e' uguale a quello attuale
            while (Directory.Exists(dataSourceDir) && dataSourcePath == default) {
               var path = Path.Combine(dataSourceDir, "Data", dataSetName);
               if (File.Exists(path)) {
                  var dataSourceBytes = File.ReadAllBytes(path);
                  if (dataSourceBytes.SequenceEqual(dataSetBytes)) {
                     dataSourcePath = path;
                     break;
                  }
               }
               dataSourceDir = Path.GetFullPath(Path.Combine(dataSourceDir, "..\\"));
            }
            // Legge le linee di dati
            var lines = new List<(string Intent, string Sentence)>();
            using (var reader = new StreamReader(new MemoryStream(dataSetBytes))) {
               for (var line = reader.ReadLine(); line != null; line = reader.ReadLine()) {
                  var tokens = line.Split('|');
                  lines.Add((Intent: tokens[0].Trim(), Sentence: tokens[1].Trim()));
               }
            }
            // Sentenza impostata
            var sentence = textBoxSentence.Text.Trim();
            // Intenzione selezionata
            var intent = cb.SelectedItem.ToString();
            // Verifica se richiesta nuova intenzione
            if (intent == "[New...]") {
               var textBox = new TextBox
               {
                  Size = new Size(200, 32),
                  Dock = DockStyle.Fill
               };
               var inputForm = new Form
               {
                  AutoSize = true,
                  AutoSizeMode = AutoSizeMode.GrowAndShrink,
                  FormBorderStyle = FormBorderStyle.None,
                  Location = PointToScreen(new Point(comboBoxIntent.Left + comboBoxIntent.Width - textBox.Size.Width, comboBoxIntent.Top + comboBoxIntent.Height)),
                  MinimumSize = textBox.Size,
                  Size = textBox.Size,
                  StartPosition = FormStartPosition.Manual,
               };
               inputForm.Controls.Add(textBox);
               //inputForm.AutoSizeMode = AutoSizeMode.GrowAndShrink;
               textBox.KeyDown += (_sender, _e) =>
               {
                  switch (_e.KeyCode) {
                     case Keys.Enter: inputForm.DialogResult = DialogResult.OK; break;
                     case Keys.Escape: inputForm.DialogResult = DialogResult.Cancel; break;
                  }
               };
               if (inputForm.ShowDialog(this) != DialogResult.OK)
                  return;
               intent = textBox.Text.Trim();
               if (string.IsNullOrWhiteSpace(intent))
                  return;
            }
            // Cerca l'indice della sentenza nel file
            var index = lines.FindIndex(item => item.Sentence == sentence);
            // Aggiunge o aggiorna la linea con la nuova intenzione programmata
            var modified = false;
            if (index > -1) {
               if (lines[index].Intent != intent) {
                  lines[index] = (Intent: cb.SelectedItem.ToString(), Sentence: sentence);
                  modified = true;
               }
            }
            else {
               lines.Add((Intent: intent, Sentence: sentence));
               modified = true;
            }
            if (modified) {
               // Aggiorna il file di dati
               using (var writer = new StreamWriter(dataSetPath)) {
                  foreach (var (Intent, Sentence) in lines)
                     writer.WriteLine($"{Intent}|{Sentence}");
               }
               // Aggiorna l'eventuale file da cui proviene il set di dati
               if (dataSourcePath != default)
                  File.Copy(dataSetPath, dataSourcePath, true);
               ml = null;
               predictor = null;
               MakePrediction();
            }
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
            if (ml == null)
               textBoxOutput.Clear();
            var dataSetPath = dataSetName != null ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
            var sentence = textBoxSentence.Text;
            var intents = default(HashSet<string>);
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
                        if (e.Kind < ChannelMessageKind.Info || cancel.IsCancellationRequested)
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
                  // Elenco delle intenzioni
                  intents = new HashSet<string>(from v in dataView.GetColumn<string>(nameof(PageIntentSdcaData.Intent)) select v.Trim());
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
            if (intents != default) {
               comboBoxIntent.Items.Clear();
               var intentsList = new List<string>(intents);
               intentsList.Sort();
               comboBoxIntent.Items.Add("[New...]");
               intentsList.ForEach(intent => comboBoxIntent.Items.Add(intent));
            }
            if (predictor != null) {
               // Aggiorna la previsione
               var prediction = predictor.Predict(new PageIntentSdcaData { Sentence = sentence });
               var intentIx = new List<string>(from object item in comboBoxIntent.Items select item.ToString()).FindIndex(item => item == prediction.Intent);
               comboBoxIntent.SelectedIndex = intentIx;
            }
            else
               comboBoxIntent.SelectedIndex = -1;
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            comboBoxIntent.SelectedIndex = -1;
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
