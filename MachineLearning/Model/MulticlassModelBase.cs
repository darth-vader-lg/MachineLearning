﻿using MachineLearning.Data;
using MachineLearning.Trainers;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i previsori di tipo multiclasse
   /// </summary>
   [Serializable]
   public abstract class MulticlassModelBase : ModelBase
   {
      #region Fields
      /// <summary>
      /// Evento di modello di autotraining disponibile
      /// </summary>
      [NonSerialized]
      private ManualResetEvent _autoTrainingModelAvailable;
      /// <summary>
      /// Coda di modelli di autotraining
      /// </summary>
      [NonSerialized]
      private Queue<(ITransformer Model, MulticlassClassificationMetrics Metrics)> _autoTrainingModels = new Queue<(ITransformer Model, MulticlassClassificationMetrics Metrics)>();
      /// <summary>
      /// Task di autotraining
      /// </summary>
      [NonSerialized]
      private readonly CancellableTask _autoTrainingTask = new CancellableTask(cancellation => Task.CompletedTask);
      #endregion
      #region Properties
      /// <summary>
      /// Metrica di scelta del miglior modello
      /// </summary>
      public MulticlassClassificationMetric BestModelSelectionMetric { get; set; }
      /// <summary>
      /// Nome colonna label
      /// </summary>
      public string LabelColumnName { get; protected set; } = "Label";
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      [field: NonSerialized]
      public MulticlassClassificationTrainersCatalog Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public MulticlassModelBase() : base() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public MulticlassModelBase(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public MulticlassModelBase(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Effettua il training con la ricerca automatica del miglior trainer
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="maxTimeInSeconds">Numero massimo di secondi di training</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni incrociate</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il modello migliore</returns>
      public override ITransformer AutoTraining(
         IDataAccess data,
         int maxTimeInSeconds,
         out object metrics,
         int numberOfFolds = 1,
         CancellationToken cancellation = default)
      {
         // Avvia il task di autotraining se necessario
         if (_autoTrainingTask.Task.IsCompleted || _autoTrainingTask.CancellationToken.IsCancellationRequested) {
            // Ottiene le pipe
            var pipes = GetPipes();
            // Coda dei modelli di training calcolati
            var queue = _autoTrainingModels = new Queue<(ITransformer Model, MulticlassClassificationMetrics Metrics)>();
            // Evento di modello disponibile
            var availableEvent = _autoTrainingModelAvailable = new ManualResetEvent(false);
            // Funzione di accodamento modelli
            void Enqueue(string trainerName, double runtimeInSeconds, ITransformer model, MulticlassClassificationMetrics metrics)
            {
               if (pipes.Output != null) {
                  var dataFirstRow = model.Transform(data.ToDataViewFiltered(row => row.Position == 0));
                  var outputTransformer = pipes.Output.Fit(dataFirstRow);
                  model = new TransformerChain<ITransformer>(model, outputTransformer);
                  metrics = (MulticlassClassificationMetrics)GetModelEvaluation(model, data);
               }
               lock (queue) {
                  ML.NET.WriteLog($"Trainer: {trainerName}\t{runtimeInSeconds:0.#} secs", Name);
                  queue.Enqueue((model, metrics));
                  availableEvent.Set();
               }
            }
            // Progress dell'autotraining
            var progress = new AutoMLProgress<MulticlassClassificationMetrics>(ML.NET, Name,
               (sender, e) =>
               {
                  cancellation.ThrowIfCancellationRequested();
                  if (e.Exception != null)
                     sender.WriteLog(e);
                  else
                     Enqueue(e.TrainerName, e.RuntimeInSeconds, e.Model, e.ValidationMetrics);
               },
               (sender, e) =>
               {
                  cancellation.ThrowIfCancellationRequested();
                  var best = (from r in e.Results where r.Exception == null select (r.Model, r.ValidationMetrics)).Best();
                  if (best != default)
                     Enqueue(e.TrainerName, e.RuntimeInSeconds, best.Model, best.Metrics);
               });
            // Avvia il task di esperimenti di autotraining
            _autoTrainingTask.StartNew(cancellation => Task.Run(() =>
            {
               // Impostazioni dell'esperimento
               var settings = new MulticlassExperimentSettings
               {
                  CancellationToken = cancellation,
                  OptimizingMetric = BestModelSelectionMetric,
                  MaxExperimentTimeInSeconds = (uint)Math.Max(0, maxTimeInSeconds)
               };
               // Crea l'esperimento
               var experiment = ML.NET.Auto().CreateMulticlassClassificationExperiment(settings);
               // Avvia
               if (numberOfFolds > 1)
                  experiment.Execute(data, (uint)Math.Max(0, numberOfFolds), LabelColumnName, null, pipes.Input, progress);
               else
                  experiment.Execute(data, LabelColumnName, null, pipes.Input, progress);
            }, cancellation), cancellation);
         }
         // Attende un risultato dal training automatico o la cancellazione
         WaitHandle.WaitAny(new[] { _autoTrainingModelAvailable, cancellation.WaitHandle });
         cancellation.ThrowIfCancellationRequested();
         // Preleva dalla coda
         lock (_autoTrainingModels) {
            // Verifica presenza elementi
            if (_autoTrainingModels.Count < 1) {
               metrics = null;
               return null;
            }
            // Preleva elemento
            var item = _autoTrainingModels.Dequeue();
            // Resetta l'evento se la coda e' vuota
            if (_autoTrainingModels.Count == 0)
               _autoTrainingModelAvailable.Reset();
            // Restituisce il risultato
            metrics = item.Metrics;
            return item.Model;
         }
      }
      /// <summary>
      /// Effettua il training con validazione incrociata del modello
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni</param>
      /// <param name="samplingKeyColumnName">Nome colonna di chiave di campionamento</param>
      /// <param name="seed">Seme per le operazioni random</param>
      /// <returns>Il modello migliore</returns>
      public override ITransformer CrossValidateTraining(
         IDataAccess data,
         out object metrics,
         int numberOfFolds = 5,
         string samplingKeyColumnName = null,
         int? seed = null)
      {
         var results = ML.NET.MulticlassClassification.CrossValidate(data, GetPipes().Merged, numberOfFolds, LabelColumnName ?? "Label", samplingKeyColumnName, seed);
         var best = (from r in results select(r.Model, r.Metrics)).Best();
         metrics = best.Metrics;
         return best.Model;
      }
      /// <summary>
      /// Funzione di restituzione della migliore fra due valutazioni modello
      /// </summary>
      /// <param name="modelEvaluation1">Prima valutazione</param>
      /// <param name="modelEvaluation2">Seconda valutazione</param>
      /// <returns>La migliore delle due valutazioni</returns>
      /// <remarks>Tenere conto che le valutazioni potrebbero essere null</remarks>
      protected override object GetBestModelEvaluation(object modelEvaluation1, object modelEvaluation2)
      {
         var best = modelEvaluation2;
         if (modelEvaluation1 is MulticlassClassificationMetrics metrics1 && modelEvaluation2 is MulticlassClassificationMetrics metrics2)
            best = metrics2.MicroAccuracy >= metrics1.MicroAccuracy && metrics2.LogLoss < metrics1.LogLoss ? modelEvaluation2 : modelEvaluation1;
         return best;
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(ITransformer model, IDataAccess data) =>
         ML.NET.MulticlassClassification.Evaluate(model.Transform(data), LabelColumnName ?? "Label");
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is not MulticlassClassificationMetrics metrics)
            return null;
         var sb = new StringBuilder();
         sb.AppendLine(metrics.ToText());
         sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
         return sb.ToString();
      }
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => Trainers = new MulticlassClassificationTrainersCatalog(ML);
      #endregion
   }
}
