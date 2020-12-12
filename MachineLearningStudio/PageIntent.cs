using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using ML.Utilities;
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
   /// Pagina di test algoritmo per la previsione delle intenzioni
   /// </summary>
   public partial class PageIntent : UserControl
   {
      #region Fields
      /// <summary>
      /// Token di cancellazione del machine learning automatico
      /// </summary>
      private CancellationTokenSource autoMLCancellation = CancellationTokenSource.CreateLinkedTokenSource(new CancellationToken(true));
      /// <summary>
      /// Set di dati
      /// </summary>
      private string dataSetName;
      /// <summary>
      /// Flag di controllo inizializzato
      /// </summary>
      private bool initialized;
      /// <summary>
      /// Array di intenzioni conosciute
      /// </summary>
      private string[] intents;
      /// <summary>
      /// Contesto ML
      /// </summary>
      private MLContext ml;
      /// <summary>
      /// Modello di apprendimento
      /// </summary>
      private ITransformer model;
      /// <summary>
      /// Previsore di piedi
      /// </summary>
      private PredictionEngine<PageIntentData, PageIntentPrediction> predictor;
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
      [Category("Behavior"), DefaultValue(true)]
      public bool SaveModel { get; set; } = true;
      /// <summary>
      /// Abilitazione utilizzo auto tuning algoritmo
      /// </summary>
      [Category("Behavior"), DefaultValue(false)]
      public bool UseAutoML { get; set; } = false;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PageIntent()
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
            if (!autoMLCancellation.IsCancellationRequested) {
               autoMLCancellation.Cancel();
               return;
            }
            // Forza il training
            ml = null;
            predictor = null;
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
            if (dataSetName == null || sender is not ComboBox cb)
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
            // Verifica se i dati sono modificati
            if (modified) {
               // Aggiorna il file di dati
               using (var writer = new StreamWriter(dataSetPath)) {
                  lines.Sort((item1, item2) => string.Compare(item1.Intent, item2.Intent));
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
      /// <param name="delay"></param>
      private void MakePrediction(TimeSpan delay = default)
      {
         try {
            // Verifica che il controllo sia inizializzato
            if (!initialized)
               return;
            // Avvia un nuovo task di previsione
            autoMLCancellation.Cancel();
            taskPrediction.cancellation.Cancel();
            if (UseAutoML)
               autoMLCancellation = new CancellationTokenSource();
            taskPrediction.cancellation = new CancellationTokenSource();
            taskPrediction.task = TaskPrediction(taskPrediction.cancellation.Token, delay);
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
      /// <returns>Il task</returns>
      private async Task TaskPrediction(CancellationToken cancel, TimeSpan delay = default)
      {
         try {
            await Task.Delay(delay, cancel);
            // Pulizia combo in caso di ricostruzione modello
            cancel.ThrowIfCancellationRequested();
            if (ml == null)
               textBoxOutput.Clear();
            // Path del set di dati in chiaro
            var dataSetPath = !string.IsNullOrWhiteSpace(dataSetName) ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
            // Path del modello
            var modelPath = !string.IsNullOrWhiteSpace(dataSetPath) ? Path.ChangeExtension(dataSetPath, "model.zip") : null;
            // Sentenza attuale
            var sentence = textBoxSentence.Text;
            // Flag di rigenerazione elementi del combo box
            var rebuildCombo = false;
            // Task di costruzione del modello
            await Task.Run(() =>
            {
               try {
                  cancel.ThrowIfCancellationRequested();
                  // Verifica che non sia gia' stato calcolato il modello
                  if (ml != null || (dataSetPath == null && modelPath == null))
                     return;
                  // Abilitazione al caricamento del modello preesistente
                  var loadModel = SaveModel && File.Exists(modelPath) && (!File.Exists(dataSetPath) || File.GetLastWriteTime(dataSetPath) < File.GetLastWriteTime(modelPath));
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
                  if (UseAutoML) {
                     try {
                        logSourceFilter = new[] { "AutoML" };
                        var dataView = ml.Data.LoadFromTextFile<PageIntentData>(
                           path: dataSetPath,
                           hasHeader: false,
                           separatorChar: '|',
                           allowQuoting: true,
                           trimWhitespace: true,
                           allowSparse: false);
                        var experimentSettings = new MulticlassExperimentSettings
                        {
                           CacheBeforeTrainer = CacheBeforeTrainer.On,
                           CancellationToken = autoMLCancellation.Token,
                           MaxExperimentTimeInSeconds = 15,
                           OptimizingMetric = MulticlassClassificationMetric.LogLoss
                        };
                        experimentSettings.Trainers.Clear();
                        experimentSettings.Trainers.Add(MulticlassClassificationTrainer.LbfgsMaximumEntropy);
                        var experiment = ml.Auto().CreateMulticlassClassificationExperiment(experimentSettings);
                        var result = experiment.Execute(trainData: dataView, labelColumnName: nameof(PageIntentData.Intent), null, null, ml.MulticlassClassificationProgress());
                        logSourceFilter = null;
                        ml.WriteLog($"Best result = {result.BestRun.TrainerName}", "AutoML");
                        ml.WriteLog(result.BestRun.ValidationMetrics.ToText(), "AutoML");
                        ml.WriteLog($"Saving the model...", "AutoML");
                        ml.Model.Save(result.BestRun.Model, dataView.Schema, modelPath);
                        ml.WriteLog($"The model is saved in {modelPath}", "AutoML");
                        loadModel = true;
                     }
                     finally {
                        logSourceFilter = null;
                     }
                  }
                  // Verifica se deve caricare un modello preesistente
                  if (loadModel) {
                     try {
                        // Carica il modello
                        ml.WriteLog($"Loading the model from {modelPath}...", nameof(PageIntent));
                        model = ml.Model.Load(modelPath, out var _);
                        ml.WriteLog("The model is loaded", nameof(PageIntent));
                        // Disabilita il caricamento dei dati in chiaro
                        loadData = false;
                     }
                     catch (Exception) {  }
                  }
                  // Verifica se deve caricare dai dati
                  if (loadData) {
                     // Dati
                     var dataView = ml.Data.LoadFromTextFile<PageIntentData>(
                        path: dataSetPath,
                        hasHeader: false,
                        separatorChar: '|',
                        allowQuoting: true,
                        allowSparse: false);
                     // Pipeline di trasformazione dei dati
                     cancel.ThrowIfCancellationRequested();
                     var dataProcessPipeline =
                        ml.Transforms.Conversion.MapValueToKey(nameof(PageIntentData.Intent), nameof(PageIntentData.Intent)).
                        Append(ml.Transforms.Text.FeaturizeText($"{nameof(PageIntentData.Sentence)}_tf", nameof(PageIntentData.Sentence))).
                        Append(ml.Transforms.CopyColumns("Features", $"{nameof(PageIntentData.Sentence)}_tf")).
                        Append(ml.Transforms.NormalizeMinMax("Features", "Features")).
                        AppendCacheCheckpoint(ml);
                     // Algoritmo di training
                     cancel.ThrowIfCancellationRequested();
                     //var trainer = ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                     //   new SdcaMaximumEntropyMulticlassTrainer.Options
                     //   {
                     //      L2Regularization = 1E-06f,
                     //      L1Regularization = 0.25f,
                     //      ConvergenceTolerance = 0.1f,
                     //      MaximumNumberOfIterations = 100,
                     //      Shuffle = false,
                     //      BiasLearningRate = 0f,
                     //      LabelColumnName = nameof(PageIntentSdcaData.Intent),
                     //      FeatureColumnName = "Features",
                     //      NumberOfThreads = Environment.ProcessorCount,
                     //   });
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
                     // Pipeline completa di training
                     var trainingPipeline =
                        dataProcessPipeline.Append(trainer).
                        Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
                     // Effettua la miglior valutazione del modello
                     cancel.ThrowIfCancellationRequested();
                     var crossValidationResults = ml.MulticlassClassification.CrossValidate(dataView, trainingPipeline, 50, nameof(PageIntentData.Intent));
                     ml.WriteLog(crossValidationResults.ToText(), "Cross validation average metrics");
                     ml.WriteLog(crossValidationResults.Best().ToText(), "Best model metrics");
                     model = crossValidationResults.Best().Model;
                     // Salva il modello
                     if (SaveModel) {
                        cancel.ThrowIfCancellationRequested();
                        ml.WriteLog($"Saving the model...", nameof(PageIntent));
                        ml.Model.Save(model, dataView.Schema, Path.ChangeExtension(dataSetPath, "model.zip"));
                        ml.WriteLog($"The model is saved in {modelPath}", nameof(PageIntent));
                     }
                  }
                  // Crea il generatore di previsioni
                  cancel.ThrowIfCancellationRequested();
                  predictor = ml.Model.CreatePredictionEngine<PageIntentData, PageIntentPrediction>(model);
                  // Estrae l'elenco di previsioni possibili
                  var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
                  predictor.OutputSchema.GetColumnOrNull("Score").Value.Annotations.GetValue("SlotNames", ref slotNames);
                  // Riempe l'array di intenti
                  intents = slotNames.GetValues().ToArray().Select(item => item.ToString()).ToArray();
                  // Forza la rigenerazione del contenuto della combo box
                  rebuildCombo = true;
               }
               catch (OperationCanceledException)
               {
                  ml = null;
                  predictor = null;
               }
               catch (Exception exc) {
                  try {
                     ml.WriteLog(exc.ToString(), nameof(PageIntent));
                  }
                  catch (Exception) { }
                  ml = null;
                  predictor = null;
                  throw;
               }
            }, cancel);
            // Ricreazione del contenuto della combo box delle previsioni
            cancel.ThrowIfCancellationRequested();
            if (rebuildCombo || (comboBoxIntent.Items.Count == 0 && intents != default)) {
               // Cancella contenuto
               comboBoxIntent.Items.Clear();
               // Aggiunge il primo oggetto per l'aggiunta di previsioni
               comboBoxIntent.Items.Add("[New...]");
               // Aggiunge la lista di previsioni ordinata alfabeticamente
               var intentsList = new List<string>(intents);
               intentsList.Sort();
               intentsList.ForEach(intent => comboBoxIntent.Items.Add(intent));
            }
            // Verifica se esiste un gestore di previsioni
            if (predictor != null && !string.IsNullOrWhiteSpace(sentence)) {
               // Aggiorna la previsione
               var prediction = predictor.Predict(new PageIntentData { Sentence = sentence });
               // Seleziona la previsione nella combo box
               var intentIx = new List<string>(from object item in comboBoxIntent.Items select item.ToString()).FindIndex(item => item == prediction.Intent);
               comboBoxIntent.SelectedIndex = intentIx;
               // Stampa i punteggi ordinati nel log
               var scores = (from ix in Enumerable.Range(0, intents.Length)
                             let score = new
                             {
                                Intent = intents[ix],
                                Score = prediction.Score[ix]
                             }
                             orderby score.Score descending
                             select score).ToList();
               ml.WriteLog("==========", nameof(PageIntent));
               ml.WriteLog(sentence, nameof(PageIntent));
               scores.ForEach(item => ml.WriteLog($"{item.Intent}: ({(int)(item.Score * 100f)})", nameof(PageIntent)));
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
      #region class PageIntentSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageIntentSettings
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
      public PageIntentSettings PageIntent { get; set; } = new PageIntentSettings();
      #endregion
   }
}
