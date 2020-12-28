using MachineLearning;
using Microsoft.ML;
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
   /// Pagina di test algoritmo per la previsione delle intenzioni
   /// </summary>
   public partial class PageTextMeaning : UserControl
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
      /// Task di previsione
      /// </summary>
      private (Task task, CancellationTokenSource cancellation) taskPrediction = (Task.CompletedTask, new CancellationTokenSource());
      /// <summary>
      /// Colore di background dei testi
      /// </summary>
      private readonly Color textBoxBackColor;
      /// <summary>
      /// Previsore di significato testi
      /// </summary>
      private PredictorTextMeaning textMeaningPredictor;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PageTextMeaning()
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
            if (e.Kind < ChannelMessageKind.Info || e.Source != textMeaningPredictor.Name)
               return;
            textMeaningPredictor.Post(() =>
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
            taskPrediction.task = TaskPrediction(textBoxDataSetName.Text.Trim(), textBoxSentence.Text.Trim(), taskPrediction.cancellation.Token, delay, forceRebuildModel);
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
            // Imposta il nome del file di dati
            textBoxDataSetName.Text = Settings.Default.PageTextMeaning.DataSetName?.Trim();
            // Crea il previsore
            textMeaningPredictor = new PredictorTextMeaning
            {
               AutoCommitData = true,
               AutoSaveModel = true,
               DataStorage = new DataStorageTextFile(Path.Combine(Environment.CurrentDirectory, "Data", textBoxDataSetName.Text)),
               ModelStorage = new ModelStorageFile(Path.Combine(Environment.CurrentDirectory, "Data", Path.ChangeExtension(textBoxDataSetName.Text, "model.zip"))),
               Name = "Predictor",
            };
            textMeaningPredictor.SetDataFormat("Label", new TextDataOptions { AllowQuoting = true, Separators = new[] { '|' } });
            // Aggancia il log
            textMeaningPredictor.ML.NET.Log += Log;
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
      /// <param name="sentence">Sentenza attuale</param>
      /// <param name="cancel">Token di cancellazione</param>
      /// <param name="delay">Ritardo dell'avvio</param>
      /// <param name="forceRebuildModel">Forza la ricostruzione del modello da zero</param>
      /// <returns>Il task</returns>
      private async Task TaskPrediction(string dataSetName, string sentence, CancellationToken cancel, TimeSpan delay = default, bool forceRebuildModel = false)
      {
         try {
            // Attende la pausa
            await Task.Delay(delay, cancel);
            // Pulizia combo in caso di ricostruzione modello
            cancel.ThrowIfCancellationRequested();
            // Rilancia o avvia il task di training
            if (forceRebuildModel) {
               await textMeaningPredictor.StopTrainingAsync();
               _ = textMeaningPredictor.StartTrainingAsync(cancel);
            }
            cancel.ThrowIfCancellationRequested();
            textBoxIntent.Text = string.IsNullOrWhiteSpace(sentence) ? "" : (await textMeaningPredictor.GetPredictionAsync(sentence, cancel)).Meaning;
            textBoxIntent.BackColor = textBoxBackColor;
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            textBoxIntent.Text = "";
            textBoxIntent.BackColor = Color.Red;
            textMeaningPredictor.ML.NET.WriteLog(exc.ToString(), nameof(TaskPrediction));
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
               if (dataSetName != Settings.Default.PageTextMeaning.DataSetName) {
                  Settings.Default.PageTextMeaning.DataSetName = dataSetName;
                  Settings.Default.Save();
               }
            }
            // Avvia una ricostruzione del modello
            MakePrediction(new TimeSpan(0, 0, 0, 0, 500), true);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Evento keydown sul text box delle intenzioni
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void textBoxIntent_KeyDown(object sender, KeyEventArgs e)
      {
         try {
            if (e.KeyCode != Keys.Enter)
               return;
            if (sender is not TextBox tb)
               return;
            if (string.IsNullOrWhiteSpace(tb.Text))
               return;
            if (string.IsNullOrWhiteSpace(textBoxSentence.Text))
               return;
            if (string.IsNullOrWhiteSpace(dataSetName))
               return;
            textMeaningPredictor.AddTrainingData(true, textBoxIntent.Text.Trim(), textBoxSentence.Text.Trim());
            MakePrediction(default, true);
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
            textBoxIntent.Text = "";
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
      #region class PageTextMeaningSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageTextMeaningSettings
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
      public PageTextMeaningSettings PageTextMeaning { get; set; } = new PageTextMeaningSettings();
      #endregion
   }
}
