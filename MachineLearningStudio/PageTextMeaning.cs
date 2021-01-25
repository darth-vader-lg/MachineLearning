using MachineLearning;
using MachineLearning.Data;
using MachineLearning.Model;
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
      /// Previsore di significato testi
      /// </summary>
      private TextMeaningRecognizer predictor;
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
            MakePrediction(default);
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
      private void Log(object sender, MachineLearningLogEventArgs e)
      {
         try {
            if (e.Kind < MachineLearningLogKind.Info || e.Source != predictor.Name)
               return;
            var (resel, SelectionStart, SelectionLength) = (textBoxOutput.SelectionStart < textBoxOutput.TextLength, textBoxOutput.SelectionStart, textBoxOutput.SelectionLength);
            textBoxOutput.AppendText(e.Message + Environment.NewLine);
            if (resel) {
               textBoxOutput.Select(SelectionStart, SelectionLength);
               textBoxOutput.ScrollToCaret();
            }
         }
         catch (Exception) { }
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
            taskPrediction.cancellation.Cancel();
            taskPrediction.cancellation = new CancellationTokenSource();
            taskPrediction.task = TaskPrediction(textBoxSentence.Text.Trim(), delay, taskPrediction.cancellation.Token);
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
            // Crea il previsore
            var context = new MachineLearningContext { SyncLogs = true };
            context.Log += Log;
            var dataSet = Path.ChangeExtension(Path.Combine(Environment.CurrentDirectory, "Data", (Settings.Default.PageTextMeaning.DataSetName ?? "").Trim()), "data");
            try {
               if (!File.Exists(dataSet))
                  dataSet = null;
            }
            catch (Exception) {
               dataSet = null;
            }
            predictor = new TextMeaningRecognizer(context)
            {
               DataStorage = dataSet == null ? null : new DataStorageTextFile(dataSet),
               ModelStorage = new ModelStorageFile(Path.ChangeExtension(dataSet, "model.zip")),
               ModelTrainer = new ModelTrainerAuto(),
               Name = "Predictor",
            };
            // Imposta il nome del file di dati
            textBoxDataSetName.Text = Settings.Default.PageTextMeaning.DataSetName?.Trim();
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
      /// <param name="sentence">Sentenza attuale</param>
      /// <param name="delay">Ritardo dell'avvio</param>
      /// <param name="cancel">Token di cancellazione</param>
      /// <returns>Il task</returns>
      private async Task TaskPrediction(string sentence, TimeSpan delay = default, CancellationToken cancel = default)
      {
         try {
            // Attende la pausa
            await Task.Delay(delay, cancel);
            cancel.ThrowIfCancellationRequested();
            // Effettua la previsione
            if (predictor.ModelStorage != null && predictor.DataStorage != null && !string.IsNullOrEmpty(sentence))
               textBoxIntent.Text = string.IsNullOrWhiteSpace(sentence) ? "" : (await predictor.GetPredictionAsync(cancel, sentence)).Meaning;
            else
               textBoxIntent.Text = "";
            textBoxIntent.BackColor = textBoxBackColor;
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
               predictor.ClearModel();
               dataSetName = null;
               tb.BackColor = Color.Red;
            }
            else {
               // Aggiorna il set di dati
               predictor.ClearModel();
               tb.BackColor = textBoxBackColor;
               dataSetName = tb.Text.Trim();
               // Salva nei settings lo stato
               if (dataSetName != Settings.Default.PageTextMeaning.DataSetName) {
                  Settings.Default.PageTextMeaning.DataSetName = dataSetName;
                  Settings.Default.Save();
                  predictor.DataStorage = new DataStorageTextFile(Path.ChangeExtension(dataSetName, "data"));
                  predictor.ModelStorage = new ModelStorageFile(Path.ChangeExtension(dataSetName, "model.zip"));
               }
            }
            // Avvia una ricostruzione del modello
            MakePrediction(new TimeSpan(0, 0, 0, 0, 500));
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
            predictor.ClearModel();
            predictor.AddTrainingData(true, textBoxIntent.Text.Trim(), textBoxSentence.Text.Trim());
            MakePrediction(default);
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
