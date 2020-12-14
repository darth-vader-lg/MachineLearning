﻿using Microsoft.ML;
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
      private TaskCompletionSource<ITransformer[]> model = new TaskCompletionSource<ITransformer[]>();
      /// <summary>
      /// Task di previsione
      /// </summary>
      private (Task task, CancellationTokenSource cancellation) taskPrediction = (Task.CompletedTask, new CancellationTokenSource());
      /// <summary>
      /// Task di training continuo
      /// </summary>
      private (Task task, CancellationTokenSource cancellation) taskTrain = (Task.CompletedTask, new CancellationTokenSource());
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
            taskPrediction.task = TaskPrediction(textBoxDataSetName.Text.Trim(), textBoxSentence.Text.Trim(), taskPrediction.cancellation.Token, delay, forceRebuildModel);
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
            if (e.Kind < ChannelMessageKind.Info || (e.Source != nameof(TaskTrain) && e.Source != nameof(TaskPrediction)))
               return;
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
            ml = new MLContext();
            ml.Log += Ml_Log;
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
            await Task.Delay(delay, cancel);
            // Pulizia combo in caso di ricostruzione modello
            cancel.ThrowIfCancellationRequested();
            if (forceRebuildModel)
               textBoxOutput.Clear();
            cancel.ThrowIfCancellationRequested();
            // Rilancia o avvia il task di training
            if (forceRebuildModel || taskTrain.task.IsCompleted) {
               // Annulla quello attuale
               taskTrain.cancellation.Cancel();
               await taskTrain.task;
               cancel.ThrowIfCancellationRequested();
               // Lo rilancia
               model = new TaskCompletionSource<ITransformer[]>();
               taskTrain.cancellation = new CancellationTokenSource();
               taskTrain.task = TaskTrain(dataSetName, taskTrain.cancellation.Token);
            }
            // Verifica se esiste un gestore di previsioni
            cancel.ThrowIfCancellationRequested();
            // Attende il modello
            var predicionModel = await model.Task;
            // Effettua la previsione
            if (predicionModel[0] != default && !string.IsNullOrWhiteSpace(sentence)) {
               var predictionDataIn = ml.Data.LoadFromEnumerable(new[] { new { Label = "?", Sentence = sentence } });
               var predictionDataOut = predicionModel[0].Transform(predictionDataIn);
               textBoxIntent.Text = predictionDataOut.GetString("PredictedLabel");
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
            ml.WriteLog(exc.ToString(), nameof(TaskPrediction));
         }
      }
      /// <summary>
      /// Task di training continuo
      /// </summary>
      /// <param name="dataSetName">Nome del set di dati</param>
      /// <param name="cancel">Token di cancellazione</param>
      /// <returns>Il task</returns>
      private Task TaskTrain(string dataSetName, CancellationToken cancel) => Task.Factory.StartNew(async () =>
      {
         try {
            // Path del set di dati in chiaro
            var dataSetPath = !string.IsNullOrWhiteSpace(dataSetName) ? Path.Combine(Environment.CurrentDirectory, "Data", dataSetName) : null;
            // Path del modello
            var modelPath = !string.IsNullOrWhiteSpace(dataSetPath) ? Path.ChangeExtension(dataSetPath, "model.zip") : null;
            // Carica modello esistente
            if (modelPath != default && File.Exists(modelPath)) {
               var m = ml.Model.Load(modelPath, out _);
               if (!model.TrySetResult(new[] { m }))
                  model.Task.Result[0] = m;
            }
            if (dataSetPath != default && File.Exists(dataSetPath)) {
               // Opzioni del trainer
               var trainerOptions = new LbfgsMaximumEntropyMulticlassTrainer.Options
               {
                  HistorySize = 1000,
                  OptimizationTolerance = 1E-06f,
                  ShowTrainingStatistics = true,
                  L2Regularization = 1E-06f,
                  L1Regularization = 0.25f,
                  MaximumNumberOfIterations = 100,
                  NumberOfThreads = 8,
               };
               // Trainer
               var trainer = ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy(trainerOptions);
               var pipe =
                  ml.Transforms.Conversion.MapValueToKey("Label").
                  Append(ml.Transforms.Text.FeaturizeText("Sentence_tf", "Sentence")).
                  Append(ml.Transforms.CopyColumns("Features", "Sentence_tf")).
                  Append(ml.Transforms.NormalizeMinMax("Features")).
                  AppendCacheCheckpoint(ml).
                  Append(trainer).
                  Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
               // Task di salvataggio del modello
               var taskSaveModel = (task: Task.CompletedTask, cancellation: CancellationTokenSource.CreateLinkedTokenSource(cancel));
               // Loop di training continuo
               var prevMetrics = default(MulticlassClassificationMetrics);
               var prevDataSetTime = File.GetLastWriteTime(dataSetPath);
               int seed = 0;
               while (!cancel.IsCancellationRequested) {
                  try {
                     // Dati di input
                     var dataView = ml.Data.LoadFromTextFile(
                        path: dataSetPath,
                        columns: new[]
                        {
                           new TextLoader.Column("Label", DataKind.String, 0),
                           new TextLoader.Column("Sentence", DataKind.String, 1),
                        },
                        hasHeader: false,
                        separatorChar: '|',
                        allowQuoting: true,
                        allowSparse: false);
                     //using var stream = new MemoryStream();
                     //ml.Data.SaveAsText(dataView, stream, '|', false, false);
                     //stream.Position = 0;
                     //using var textReader = new StreamReader(stream);
                     //var dic = new System.Collections.Generic.Dictionary<string, string>();
                     //for (var line = textReader.ReadLine(); line != default; line = textReader.ReadLine()) {
                     //   var items = line.Split('|', 2);
                     //   dic[items[1]] = items[0];
                     //}


                     //// Effettua il training
                     //var model = pipe.Fit(dataView);
                     //// Valuta
                     //var metrics = ml.MulticlassClassification.Evaluate(model.Transform(dataView));

                     var crossValidation = ml.MulticlassClassification.CrossValidate(dataView, pipe, 10, "Label", null, seed++);
                     var best = crossValidation.Best();
                     var model = best.Model;
                     var metrics = best.Metrics;

                     cancel.ThrowIfCancellationRequested();
                     var dataSetTime = File.GetLastWriteTime(dataSetPath);
                     // Verifica se c'e' un miglioramento; se affermativo salva il nuovo modello
                     if (prevMetrics == default || dataSetTime > prevDataSetTime || (metrics.MicroAccuracy >= prevMetrics.MicroAccuracy && metrics.LogLoss < prevMetrics.LogLoss)) {
                        // Emette il log
                        ml.WriteLog("Found best model", nameof(TaskTrain));
                        ml.WriteLog(metrics.ToText(), nameof(TaskTrain));
                        cancel.ThrowIfCancellationRequested();
                        // Annulla eventuali task di salvataggio precedenti
                        taskSaveModel.cancellation.Cancel();
                        await taskSaveModel.task;
                        cancel.ThrowIfCancellationRequested();
                        // Avvia il task di salvataggio
                        taskSaveModel.cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancel);
                        var savingModel = model;
                        taskSaveModel.task = Task.Run(() =>
                        {
                           taskSaveModel.cancellation.Token.ThrowIfCancellationRequested();
                           ml.Model.Save(savingModel, dataView.Schema, modelPath);
                        }, taskSaveModel.cancellation.Token);
                        prevMetrics = metrics;
                        prevDataSetTime = dataSetTime;
                        // Aggiorna il modello attuale
                        if (!this.model.TrySetResult(new[] { model }))
                           this.model.Task.Result[0] = model;
                     }
                  }
                  catch (OperationCanceledException) { }
                  catch (Exception exc) {
                     Trace.WriteLine(exc);
                     ml.WriteLog(exc.Message, nameof(TaskTrain));
                  }
               }
            }
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            ml.WriteLog(exc.ToString(), nameof(TaskTrain));
         }
         if (!model.TrySetResult(new ITransformer[] { default }))
            model.Task.Result[0] = default;
      }, cancel, TaskCreationOptions.LongRunning, TaskScheduler.Default);
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
            if (sender is not TextBox tb || string.IsNullOrWhiteSpace(tb.Text) || e.KeyCode != Keys.Enter)
               return;
            //// Avvia un nuovo task di retraining
            //taskRetrain.cancellation.Cancel();
            //taskRetrain.cancellation = new CancellationTokenSource();
            //taskRetrain.task = TaskRetrain(taskRetrain.cancellation.Token);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            throw;
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
