using MachineLearning.Data;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i modelli ML.NET
   /// </summary>
   [Serializable]
   public abstract partial class ModelBaseMLNet : ModelBase, IContextProvider<MLContext>, IModelTrainingShuffle, IModelTrainingStandard, ITransformer
   {
      #region Properties
      /// <summary>
      /// Contesto
      /// </summary>
      public MLContext Context => ((IContextProvider<MLContext>)GetChannelProvider()).Context; 
      /// <summary>
      /// Indica se una chiamata alla GetRowToRowMapper avra successo con lo schema appropriato
      /// </summary>
      public bool IsRowToRowMapper => (GetEvaluation(new ModelTrainerStandard()).Model as ITransformer)?.IsRowToRowMapper ?? false;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      public ModelBaseMLNet(IContextProvider<MLContext> contextProvider = default) : base(contextProvider ?? MachineLearningContext.Default) { }
      /// <summary>
      /// Restituisce lo schema di output dato lo schema di input
      /// </summary>
      /// <param name="inputSchema">Scema di input</param>
      /// <returns>Lo schema di output</returns>
      public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => (GetEvaluation(new ModelTrainerStandard()).Model as ITransformer)?.GetOutputSchema(inputSchema);
      /// <summary>
      /// Restituisce le pipe di training del modello
      /// </summary>
      /// <returns>Le pipe</returns>
      public abstract ModelPipes GetPipes();
      /// <summary>
      /// Restituisce il mapper riga a riga
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il mappatore</returns>
      public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => (GetEvaluation(new ModelTrainerStandard()).Model as ITransformer)?.GetRowToRowMapper(inputSchema);
      /// <summary>
      /// Restituisce il modello sottoposto al training
      /// </summary>
      /// <param name="trainer">Il trainer da utilizzare</param>
      /// <param name="data">Dati di training</param>
      /// <param name="metrics">Eventuale metrica</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns></returns>
      protected sealed override IDataTransformer GetTrainedModel(IModelTrainer trainer, IDataAccess data, out object metrics, CancellationToken cancellation)
      {
         Channel.CheckValue(trainer, nameof(trainer));
         Channel.CheckValue(data, nameof(data));
         return trainer.GetTrainedModel(this, data, out metrics, cancellation);
      }
      /// <summary>
      /// Restituisce il modello effettuando il training con shuffle
      /// </summary>
      /// <param name="data">Dati di training</param>
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      IDataTransformer IModelTrainingShuffle.ShuffleTraining(IDataAccess data, out object evaluationMetrics, int? seed, CancellationToken cancellation)
      {
         evaluationMetrics = null;
         var result = GetPipes().Merged.Fit(data.CanShuffle ? Context.Data.ShuffleRows(data, seed) : data);
         cancellation.ThrowIfCancellationRequested();
         return result == null ? null : new DataTransformerMLNet(this, result);
      }
      /// <summary>
      /// Restituisce il modello effettuando il training standard
      /// </summary>
      /// <param name="data">Dati di training</param>
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      IDataTransformer IModelTrainingStandard.StandardTraining(IDataAccess data, out object evaluationMetrics, CancellationToken cancellation)
      {
         evaluationMetrics = null;
         var result = GetPipes().Merged.Fit(data);
         cancellation.ThrowIfCancellationRequested();
         return result == null ? null : new DataTransformerMLNet(this, result);
      }
      /// <summary>
      /// Carica i dati da uno storage
      /// </summary>
      /// <param name="dataStorage">Storage di dati</param>
      /// <returns>La vista di dati</returns>
      public sealed override IDataAccess LoadData(IDataStorage dataStorage)
      {
         Channel.CheckValue(dataStorage, nameof(dataStorage));
         var options = (this as ITextLoaderOptions)?.TextLoaderOptions ?? new TextLoader.Options() { Columns = (this as IInputSchema)?.InputSchema?.ToTextLoaderColumns() };
         return dataStorage.LoadData(Context, options);
      }
      /// <summary>
      /// Carica il modello da uno storage
      /// </summary>
      /// <param name="modelStorage">Storage del modello</param>
      /// <param name="schema">Lo schema del modello</param>
      /// <returns>Il modello</returns>
      public sealed override IDataTransformer LoadModel(IModelStorage modelStorage, out DataViewSchema schema)
      {
         Channel.CheckValue(modelStorage, nameof(modelStorage));
         return new DataTransformerMLNet(this, modelStorage.LoadModel(Context, out schema));
      }
      /// <summary>
      /// Effettua il salvataggio del modello
      /// </summary>
      /// <param name="ctx">Contesto di salvataggio</param>
      void ICanSaveModel.Save(ModelSaveContext ctx)
      {
         var evaluation = GetEvaluation(new ModelTrainerStandard());
         if (evaluation.Model != null && evaluation.ModelStorage != null && evaluation.Model is ITransformer transformer)
            evaluation.ModelStorage.SaveModel(Context, transformer, evaluation.InputSchema);
      }
      /// <summary>
      /// Salva i dati in uno storage
      /// </summary>
      /// <param name="dataStorage">Storage di dati</param>
      /// <param name="data">Dati</param>
      public sealed override void SaveData(IDataStorage dataStorage, IDataAccess data)
      {
         Channel.CheckValue(dataStorage, nameof(dataStorage));
         Channel.CheckValue(data, nameof(data));
         var options = (this as ITextLoaderOptions)?.TextLoaderOptions ?? new TextLoader.Options() { Columns = (this as IInputSchema)?.InputSchema?.ToTextLoaderColumns() };
         dataStorage.SaveData(Context, data, options);
      }
      /// <summary>
      /// Salva il modello in uno storage
      /// </summary>
      /// <param name="modelStorage">Storage del modello</param>
      /// <param name="model">Modello</param>
      /// <param name="schema">Lo schema del modello</param>
      /// <returns>Il modello</returns>
      public sealed override void SaveModel(IModelStorage modelStorage, IDataTransformer model, DataViewSchema schema)
      {
         Channel.CheckValue(modelStorage, nameof(modelStorage));
         Channel.CheckValue(model, nameof(model));
         var transformer = model as ITransformer;
         Channel.Check(transformer != null, $"The model doesn't implement the {typeof(ITransformer)} interface");
         modelStorage.SaveModel(Context, transformer, schema);
      }
      /// <summary>
      /// Trasforma i dati di input per il modello
      /// </summary>
      /// <param name="input">Vista di dati di input</param>
      /// <returns>I dati trasformati</returns>
      public IDataView Transform(IDataView input) => (GetEvaluation(new ModelTrainerStandard()).Model as ITransformer)?.Transform(input);
      #endregion
   }

   /// <summary>
   /// Helper di autotraining
   /// </summary>
   partial class ModelBaseMLNet // AutoTrainingTask
   {
      internal protected class AutoTrainingTask<TMetrics, TExperimentSettings> : IDisposable where TMetrics : class where TExperimentSettings : ExperimentSettings
      {
         #region Fields
         /// <summary>
         /// Evento di modello di autotraining disponibile
         /// </summary>
         private ManualResetEvent autoTrainingModelAvailable;
         /// <summary>
         /// Coda di modelli di autotraining
         /// </summary>
         private Queue<(DataTransformerMLNet Model, TMetrics Metrics)> autoTrainingModels = new Queue<(DataTransformerMLNet Model, TMetrics Metrics)>();
         /// <summary>
         /// Task di autotraining
         /// </summary>
         private readonly CancellableTask autoTrainingTask = new CancellableTask(cancellation => Task.CompletedTask);
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly ModelBaseMLNet owner;
         #endregion
         #region Properties
         /// <summary>
         /// Stato di running del task
         /// </summary>
         public bool IsRunning => !autoTrainingTask.Task.IsCompleted && !autoTrainingTask.CancellationToken.IsCancellationRequested;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         public AutoTrainingTask(ModelBaseMLNet owner) => this.owner = owner;
         /// <summary>
         /// Finalizzatore
         /// </summary>
         ~AutoTrainingTask() => Dispose(disposing: false);
         /// <summary>
         /// Dispose da programma
         /// </summary>
         public void Dispose()
         {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
         }
         /// <summary>
         /// Funzione di dispose
         /// </summary>
         /// <param name="disposing"></param>
         protected virtual void Dispose(bool disposing)
         {
            if (disposing) {
               if (!autoTrainingTask.Task.IsCompleted) {
                  autoTrainingTask.Task.ContinueWith(t =>
                  {
                     try {
                        autoTrainingModelAvailable?.Dispose();
                        autoTrainingModelAvailable = null;
                     }
                     catch (Exception exc) {
                        Trace.WriteLine(exc);
                     }
                  });
               }
               else {
                  try {
                     autoTrainingModelAvailable?.Dispose();
                     autoTrainingModelAvailable = null;
                  }
                  catch (Exception exc) {
                     Trace.WriteLine(exc);
                  }
               }
            }
            else if (autoTrainingTask.Task.IsCompleted) {
               autoTrainingModelAvailable?.Dispose();
               autoTrainingModelAvailable = null;
            }
         }
         /// <summary>
         /// Effettua il training con la ricerca automatica del miglior trainer
         /// </summary>
         /// <param name="ExperimentCreator">Creatore di esperimenti</param>
         /// <param name="BestModelEvaluator">Evaluator del miglior modello</param>
         /// <param name="data">Dati</param>
         /// <param name="labelColumnName">Nome colonna label</param>
         /// <param name="metrics">La metrica del modello migliore</param>
         /// <param name="numberOfFolds">Numero di validazioni incrociate</param>
         /// <param name="cancellation">Token di cancellazione</param>
         /// <returns>Il modello migliore</returns>
         public IDataTransformer WaitResult(
            Func<ExperimentBase<TMetrics, TExperimentSettings>> ExperimentCreator,
            Func<IEnumerable<(ITransformer Model, TMetrics Metrics)>, (ITransformer Model, TMetrics Metrics)> BestModelEvaluator,
            IDataAccess data,
            string labelColumnName,
            out TMetrics metrics,
            int numberOfFolds = 1,
            CancellationToken cancellation = default)
         {
            // Avvia il task di autotraining se necessario
            if (autoTrainingModels.Count == 0 && (autoTrainingTask.Task.IsCompleted || autoTrainingTask.CancellationToken.IsCancellationRequested)) {
               // Ottiene le pipe
               var pipes = owner.GetPipes();
               // Coda dei modelli di training calcolati
               var queue = autoTrainingModels = new Queue<(DataTransformerMLNet Model, TMetrics Metrics)>();
               // Evento di modello disponibile
               var availableEvent = autoTrainingModelAvailable = new ManualResetEvent(false);
               // Funzione di accodamento modelli
               void Enqueue(string trainerName, double runtimeInSeconds, DataTransformerMLNet model, TMetrics metrics)
               {
                  try {
                     if (model != null && pipes.Output != null) {
                        var dataFirstRow = model.Transform(data.ToDataViewFiltered(row => row.Position == 0), cancellation);
                        var outputTransformer = pipes.Output.Fit(dataFirstRow);
                        model = new DataTransformerMLNet(owner, new TransformerChain<ITransformer>(model.Transformer, outputTransformer));
                        metrics = (TMetrics)owner.GetModelEvaluation(model, data);
                     }
                     lock (queue) {
                        owner.Channel.WriteLog(model == null ? $"Autotraining complete" : $"Trainer: {trainerName}\t{runtimeInSeconds:0.#} secs");
                        queue.Enqueue((model, metrics));
                        availableEvent.Set();
                     }
                  }
                  catch (Exception exc) {
                     Trace.WriteLine(exc);
                     try {
                       owner.Channel.WriteLog(exc.ToString());
                     }
                     catch (Exception) {
                     }
                  }
               }
               // Progress dell'autotraining
               var progress = new AutoMLProgress<TMetrics>(owner,
                  (sender, e) =>
                  {
                     cancellation.ThrowIfCancellationRequested();
                     if (e.Exception != null)
                        sender.WriteLog(e);
                     else
                        Enqueue(e.TrainerName, e.RuntimeInSeconds, new DataTransformerMLNet(owner, e.Model), e.ValidationMetrics);
                  },
                  (sender, e) =>
                  {
                     cancellation.ThrowIfCancellationRequested();
                     var best = BestModelEvaluator(from r in e.Results where r.Exception == null select (r.Model, r.ValidationMetrics));
                     if (best != default)
                        Enqueue(e.TrainerName, e.RuntimeInSeconds, new DataTransformerMLNet(owner, best.Model), best.Metrics);
                  });
               // Avvia il task di esperimenti di autotraining
               autoTrainingTask.StartNew(cancellation => Task.Factory.StartNew(() =>
               {
                  // Impostazioni dell'esperimento
                  try {
                     // Crea l'esperimento
                     var experiment = ExperimentCreator();
                     // Avvia
                     if (numberOfFolds > 1)
                        experiment.Execute(data, (uint)Math.Max(0, numberOfFolds), labelColumnName, null, pipes.Input, progress);
                     else
                        experiment.Execute(data, labelColumnName, null, pipes.Input, progress);
                  }
                  catch (Exception exc) {
                     Trace.WriteLine(exc);
                     owner.Channel.WriteLog(exc.ToString());
                  }
                  Enqueue(null, 0.0, null, default);
               },
               cancellation,
               TaskCreationOptions.LongRunning,
               TaskScheduler.Default), cancellation);
            }
            // Attende un risultato dal training automatico o la cancellazione
            var waitResult = WaitHandle.WaitAny(new[] { autoTrainingModelAvailable, cancellation.WaitHandle });
            cancellation.ThrowIfCancellationRequested();
            // Preleva dalla coda
            lock (autoTrainingModels) {
               // Verifica presenza elementi
               if (autoTrainingModels.Count < 1) {
                  metrics = default;
                  return null;
               }
               // Preleva elemento
               var item = autoTrainingModels.Dequeue();
               // Resetta l'evento se la coda e' vuota
               if (autoTrainingModels.Count == 0)
                  autoTrainingModelAvailable.Reset();
               // Restituisce il risultato
               metrics = item.Metrics;
               return item.Model;
            }
         }
         #endregion
      }
   }
}
