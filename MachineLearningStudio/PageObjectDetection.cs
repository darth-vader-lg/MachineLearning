using MachineLearning;
using MachineLearning.Model;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MachineLearningStudio
{
   /// <summary>
   /// Pagina di test algoritmo di rilevamento oggetti nelle immagini
   /// </summary>
   public partial class PageObjectDetection : UserControl
   {
      #region Fields
      /// <summary>
      /// Flag di controllo inizializzato
      /// </summary>
      private bool initialized;
      /// <summary>
      /// Previsore di oggetti
      /// </summary>
      private ObjectDetection predictor;
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
      public PageObjectDetection()
      {
         InitializeComponent();
         textBoxBackColor = textBoxModelDir.BackColor;
      }
      /// <summary>
      /// Pulsante di browsing del modello
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void buttonBrowseModel_Click(object sender, EventArgs e)
      {
         using var fbd = new FolderBrowserDialog(); DialogResult result = fbd.ShowDialog();
         if (result == DialogResult.OK && !string.IsNullOrWhiteSpace(fbd.SelectedPath))
            textBoxModelDir.Text = fbd.SelectedPath;
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
      /// Marca l'immagine
      /// </summary>
      /// <param name="bmp">Bitmap da marcare</param>
      /// <param name="rect">Rettangolo da disegnare</param>
      /// <param name="score">Punteggio della previsione</param>
      /// <param name="name">Nome della previsione</param>
      private static void DrawObjectOnBitmap(Bitmap bmp, ObjectDetection.Prediction.Box box)
      {
         using var graphic = Graphics.FromImage(bmp); graphic.SmoothingMode = SmoothingMode.AntiAlias;
         var rect = new Rectangle(
            (int)(box.Left * bmp.Size.Width),
            (int)(box.Top * bmp.Height),
            (int)(box.Width * bmp.Width),
            (int)(box.Height * bmp.Height));
         using var pen = new Pen(Color.Lime, Math.Max(Math.Min(rect.Width, rect.Height) / 320f, 1f));
         graphic.DrawRectangle(pen, rect);
         var fontSize = Math.Min(bmp.Size.Width, bmp.Size.Height) / 40f;
         fontSize = Math.Max(fontSize, 8f);
         fontSize = Math.Min(fontSize, rect.Height);
         using var font = new Font("Verdana", fontSize, GraphicsUnit.Pixel);
         var p = new Point(rect.Left, rect.Top);
         var text = $"{box.Name}:{(int)(box.Score * 100)}";
         var size = graphic.MeasureString(text, font);
         using var brush = new SolidBrush(Color.FromArgb(50, Color.Lime));
         graphic.FillRectangle(brush, p.X, p.Y, size.Width, size.Height);
         graphic.DrawString(text, font, Brushes.Black, p);
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
            var modelDir = Settings.Default.PageObjectDetection.ModelDir ?? "".Trim();
            try {
               if (!Directory.Exists(modelDir))
                  modelDir = null;
            }
            catch (Exception) {
               modelDir = null;
            }
            predictor = new ObjectDetection(context)
            {
               ModelStorage = modelDir == null ? null : new ModelStorageFile(modelDir),
               Name = "Predictor",
            };
            textBoxModelDir.Text = Settings.Default.PageObjectDetection.ModelDir?.Trim();
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
               if (prediction.Boxes.Length > 0) {
                  labelClassResult.Text = $"{prediction.Boxes[0].Name} ({prediction.Boxes[0].Score * 100f:0.#}%)";
                  var bmp = new Bitmap(imagePath);
                  foreach (var box in prediction.Boxes)
                     DrawObjectOnBitmap(bmp, box);
                  pictureBox.Image = bmp;
               }
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
      /// Evento di variazione del testo del path del modello
      /// </summary>
      /// <param name="sender"></param>
      /// <param name="e"></param>
      private void textBoxModelDir_TextChanged(object sender, EventArgs e)
      {
         try {
            if (sender is not TextBox tb)
               return;
            var path = tb.Text.Trim();
            if (!Directory.Exists(path)) {
               predictor.ClearModel();
               tb.BackColor = Color.Red;
            }
            else {
               predictor.ClearModel();
               tb.BackColor = textBoxBackColor;
               var modelDir = tb.Text.Trim();
               if (modelDir != Settings.Default.PageObjectDetection.ModelDir) {
                  Settings.Default.PageObjectDetection.ModelDir = modelDir;
                  Settings.Default.Save();
                  predictor.ModelStorage = new ModelStorageFile(modelDir);
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
      #region class PageObjectDetectionSettings
      /// <summary>
      /// Impostazione della pagina
      /// </summary>
      [Serializable]
      public class PageObjectDetectionSettings
      {
         #region Properties
         /// <summary>
         /// Directory del modello
         /// </summary>
         public string ModelDir { get; set; } = "";
         #endregion
      }
      #endregion
      #region Properties
      /// <summary>
      /// Settings della pagina
      /// </summary>
      public PageObjectDetectionSettings PageObjectDetection { get; set; } = new PageObjectDetectionSettings();
      #endregion
   }
}
