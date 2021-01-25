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
   /// Pagina di test algoritmo di classificazione immagini
   /// </summary>
   public partial class PageImageClassification : UserControl
   {
      #region Fields
      /// <summary>
      /// Flag di controllo inizializzato
      /// </summary>
      private bool initialized;
      /// <summary>
      /// Previsore di immagini
      /// </summary>
      private ImageRecognizer predictor;
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
      public PageImageClassification()
      {
         InitializeComponent();
         textBoxBackColor = textBoxImageSetName.BackColor;
      }
      /// <summary>
      /// Click sul pulsante di loading dell'immagine
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void buttonLoad_Click(object sender, EventArgs e)
      {
         try {
            if (openFileDialog.ShowDialog(this) != DialogResult.OK)
               return;
            labelClassResult.Text = "";
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
            if (sender is not CheckBox cb)
               return;
            var crossValidate = cb.Checked;
            if (crossValidate != Settings.Default.PageImageClassification.CrossValidate) {
               try {
                  predictor.ClearModel();
                  var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", Path.ChangeExtension(textBoxImageSetName.Text.Trim(), "model.zip"));
                  if (File.Exists(modelPath))
                     File.Delete(modelPath);
                  predictor.ModelTrainer = crossValidate ? new ModelTrainerCrossValidation { NumFolds = 5 } : new ModelTrainerStandard();
               }
               catch  { }
               Settings.Default.PageImageClassification.CrossValidate = crossValidate;
               Settings.Default.Save();
            }
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
            if ((e.Kind < MachineLearningLogKind.Info || e.Source == "TextSaver; Saving") && !e.Source.Contains("Trainer"))
               return;
            var (resel, SelectionStart, SelectionLength) = (textBoxOutput.SelectionStart < textBoxOutput.TextLength, textBoxOutput.SelectionStart, textBoxOutput.SelectionLength);
            textBoxOutput.AppendText(e.Message + Environment.NewLine);
            if (resel) {
               textBoxOutput.Select(SelectionStart, SelectionLength);
               textBoxOutput.ScrollToCaret();
            }
         }
         catch (Exception) {
         }
      }
      /// <summary>
      /// Effettua la previsione in base ai dati impostati
      /// </summary>
      private void MakePrediction()
      {
         try {
            // Verifica che il controllo sia inizializzato
            if (!initialized)
               return;
            // Avvia un nuovo task di previsione
            taskPrediction.cancellation.Cancel();
            taskPrediction.cancellation = new CancellationTokenSource();
            taskPrediction.task = TaskPrediction(openFileDialog.FileName, taskPrediction.cancellation.Token);
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
            var dataSetDir = Path.Combine(Environment.CurrentDirectory, "Data", (Settings.Default.PageImageClassification.DataSetDir ?? "").Trim());
            try {
               if (!Directory.Exists(dataSetDir))
                  dataSetDir = null;
            }
            catch (Exception) {
               dataSetDir = null;
            }
            predictor = new ImageRecognizer(context)
            {
               ImagesSources = dataSetDir == null ? null : new[] { dataSetDir },
               DataStorage = dataSetDir == null ? null : new DataStorageTextFile(Path.ChangeExtension(dataSetDir, "data")),
               ModelStorage = dataSetDir == null ? null : new ModelStorageFile(Path.ChangeExtension(dataSetDir, "model.zip")),
               ModelTrainer = Settings.Default.PageImageClassification.CrossValidate ? new ModelTrainerCrossValidation { NumFolds = 5 } : new ModelTrainerStandard(),
               Name = "Predictor",
            };
            textBoxImageSetName.Text = Settings.Default.PageImageClassification.DataSetDir?.Trim();
            checkBoxCrossValidate.Checked = Settings.Default.PageImageClassification.CrossValidate;
            initialized = true;
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Task di previsione
      /// </summary>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task</returns>
      private async Task TaskPrediction(string imagePath, CancellationToken cancellation)
      {
         try {
            cancellation.ThrowIfCancellationRequested();
            if (predictor.ModelStorage != null && !string.IsNullOrEmpty(imagePath) && File.Exists(imagePath)) {
               var prediction = await Task.Run(() => predictor.GetPredictionAsync(imagePath, cancellation));
               labelClassResult.Text = $"{prediction.Kind} ({prediction.Score * 100f:0.#}%)";
            }
            else
               labelClassResult.Text = "";
            labelClassResult.BackColor = textBoxBackColor;
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            labelClassResult.Text = "";
            labelClassResult.BackColor = Color.Red;
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
            if (sender is not TextBox tb)
               return;
            var path = Path.Combine(Environment.CurrentDirectory, "Data", tb.Text.Trim());
            if (!Directory.Exists(path)) {
               predictor.ClearModel();
               tb.BackColor = Color.Red;
               predictor.ImagesSources = null;
            }
            else {
               predictor.ClearModel();
               tb.BackColor = textBoxBackColor;
               var dataSetDir = tb.Text.Trim();
               if (dataSetDir != Settings.Default.PageImageClassification.DataSetDir) {
                  Settings.Default.PageImageClassification.DataSetDir = dataSetDir;
                  Settings.Default.Save();
                  predictor.ImagesSources = new[] { path };
                  predictor.DataStorage = new DataStorageTextFile(Path.ChangeExtension(path, "data"));
                  predictor.ModelStorage = new ModelStorageFile(Path.ChangeExtension(path, "model.zip"));
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
