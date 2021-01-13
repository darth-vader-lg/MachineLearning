using MachineLearning;
using MachineLearning.Data;
using MachineLearning.Model;
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
      private ImagesRecognizer predictor;
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
                  var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", Path.ChangeExtension(textBoxImageSetName.Text.Trim(), "model.zip"));
                  if (File.Exists(modelPath))
                     File.Delete(modelPath);
               }
               catch  { }
               Settings.Default.PageImageClassification.CrossValidate = crossValidate;
               Settings.Default.Save();
            }
            MakePrediction(true);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Effettua la previsione in base ai dati impostati
      /// </summary>
      /// <param name="rebuildModel">Forzatura ricostruzione modello</param>
      private void MakePrediction(bool rebuildModel = false)
      {
         try {
            // Verifica che il controllo sia inizializzato
            if (!initialized)
               return;
            // Avvia un nuovo task di previsione
            taskPrediction.cancellation.Cancel();
            taskPrediction.cancellation = new CancellationTokenSource();
            taskPrediction.task = TaskPrediction(textBoxImageSetName.Text.Trim(), openFileDialog.FileName, checkBoxCrossValidate.Checked, rebuildModel, taskPrediction.cancellation.Token);
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
      private void Ml_Log(object sender, LoggingEventArgs e)
      {
         try {
            if ((e.Kind < ChannelMessageKind.Info || e.Source == "TextSaver; Saving") && !e.Source.Contains("Trainer"))
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
      /// Funzione di caricamento del controllo
      /// </summary>
      /// <param name="e"></param>
      protected override void OnLoad(EventArgs e)
      {
         // Metodo base
         try {
            base.OnLoad(e);
            // Crea il previsore
            predictor = new ImagesRecognizer { AutoSaveModel = true, Name = "Predictor", DataStorage = null };
            predictor.ML.NET.Log += Ml_Log;
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
      /// <param name="dataSetName">Nome del set di dati</param>
      /// <param name="imagePath">Path dell'immagine da classificare</param>
      /// <param name="crossValidation">Abilita la validazione incrociata</param>
      /// <param name="rebuildModel">Forzatura ricostruzione completa modello</param>
      /// <param name="cancel">Token di cancellazione</param>
      /// <returns>Il task</returns>
      private async Task TaskPrediction(string dataSetName, string imagePath, bool crossValidation, bool rebuildModel, CancellationToken cancel)
      {
         try {
            var dataStoragePath = Path.Combine(Environment.CurrentDirectory, "data", Path.ChangeExtension(dataSetName, ".data"));
            if (rebuildModel || predictor.DataStorage == null || dataStoragePath.ToLower() != ((DataStorageTextFile)predictor.DataStorage).FilePath.ToLower()) {
               await Task.Run(async () =>
               {
                  await predictor.StopTrainingAsync(cancel);
                  cancel.ThrowIfCancellationRequested();
                  predictor.DataStorage = new DataStorageTextFile(dataStoragePath);
                  await predictor.UpdateStorageAsync(Path.ChangeExtension(dataStoragePath, null), cancel);
                  predictor.ModelStorage = new ModelStorageFile(Path.Combine(Environment.CurrentDirectory, "Data", Path.ChangeExtension(dataStoragePath, "model.zip")));
                  cancel.ThrowIfCancellationRequested();
                  predictor.ValidationLevel = crossValidation ? 5 : 1;
               }, cancel);
            }
            cancel.ThrowIfCancellationRequested();
            if (!string.IsNullOrWhiteSpace(imagePath) && File.Exists(imagePath)) {
               var prediction = await Task.Run(() => predictor.GetPredictionAsync($"\"{imagePath}\"", cancel));
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
            if (!Directory.Exists(path))
               tb.BackColor = Color.Red;
            else {
               tb.BackColor = textBoxBackColor;
               var dataSetDir = tb.Text.Trim();
               if (dataSetDir != Settings.Default.PageImageClassification.DataSetDir) {
                  Settings.Default.PageImageClassification.DataSetDir = dataSetDir;
                  Settings.Default.Save();
                  MakePrediction();
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
