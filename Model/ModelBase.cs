using MachineLearning.Data;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i predittori
   /// </summary>
   [Serializable]
   public abstract partial class ModelBase : ChannelProvider, IDataTransformer, IModelTrainingControl
   {
      #region Fields
      /// <summary>
      /// Contesto
      /// </summary>
      private readonly IChannelProvider context;
      /// <summary>
      /// Valutazione
      /// </summary>
      [NonSerialized]
      private Evaluator evaluator;
      #endregion
      #region Events
      /// <summary>
      /// Evento di variazione modello
      /// </summary>
      public event ModelTrainingEventHandler ModelChanged;
      /// <summary>
      /// Evento di segnalazione ciclo di training avviato
      /// </summary>
      public event ModelTrainingEventHandler TrainingCycleStarted;
      /// <summary>
      /// Evento di variazione dati di training
      /// </summary>
      public event ModelTrainingEventHandler TrainingDataChanged;
      /// <summary>
      /// Evento di segnalazione training terminato
      /// </summary>
      public event ModelTrainingEventHandler TrainingEnded;
      /// <summary>
      /// Evento di segnalazione training avviato
      /// </summary>
      public event ModelTrainingEventHandler TrainingStarted;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      public ModelBase(IChannelProvider context)
      {
         Contracts.CheckValue(context, nameof(context));
         this.context = context;
         if (context is MachineLearningContext ml)
            ml.AddDisposable(this);
      }
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      /// <returns>Il task</returns>
      public void AddTrainingData(IDataAccess data, bool checkForDuplicates, CancellationToken cancellation = default)
      {
         if (!checkForDuplicates) {
            // Ottiene e verifica l'evaluator
            var evaluator = this.evaluator;
            var trainingStorage = evaluator != null ? evaluator.TrainingStorage : (this as ITrainingStorageProvider)?.TrainingStorage;
            if (trainingStorage == null)
               throw new InvalidOperationException("The object doesn't have training data characteristics");
            // Effettua l'accodamento
            var currentTrainingData = LoadData(trainingStorage);
            if (currentTrainingData != null) {
               var merged = currentTrainingData.Merge(new DataAccess(this, data));
               var temp = new DataStorageBinaryTempFile();
               SaveData(temp, merged);
               SaveData(trainingStorage, LoadData(temp));
            }
            else
               SaveData(trainingStorage, data);
            cancellation.ThrowIfCancellationRequested();
         }
         else {
            // Ottiene e verifica l'evaluator
            var evaluator = this.evaluator;
            var trainingStorage = evaluator != null ? evaluator.TrainingStorage : (this as ITrainingStorageProvider)?.TrainingStorage;
            if (trainingStorage == null)
               throw new InvalidOperationException("The object doesn't have training data characteristics");
            // Set di righe duplicate di training
            var invalidRows = new HashSet<long>();
            // Dati di training attuali
            var currentTrainingData = LoadData(trainingStorage);
            var currentStorageData = evaluator.DataStorage == null ? null : LoadData(evaluator.DataStorage);
            // Loop su tutti i dati di training
            var rowsCount = 0;
            foreach (var dataCursor in data.GetRowCursor(data.Schema).AsEnumerable()) {
               // Ottiene i valori
               var dataRow = dataCursor.ToDataViewValuesRow(this);
               rowsCount++;
               // Loop sui set di dati
               foreach (var dataSet in new[] { currentStorageData, currentTrainingData }) {
                  // Loop sui dati di training gia' esistenti
                  if (dataSet != null) {
                     foreach (var dataSetCursor in dataSet.GetRowCursor(dataSet.Schema).AsEnumerable()) {
                        var dataSetRow = dataSetCursor.ToDataViewValuesRow(this);
                        if (dataRow.Zip(dataSetRow).All(item => item.First == item.Second)) {
                           invalidRows.Add(dataRow.Position);
                           break;
                        }
                        cancellation.ThrowIfCancellationRequested();
                     }
                     if (invalidRows.Contains(dataRow.Position))
                        continue;
                  }
               }
            }
            // Verifica se deve aggiornare i dati di training
            cancellation.ThrowIfCancellationRequested();
            if (invalidRows.Count < rowsCount) {
               if (currentTrainingData != null) {
                  var merged = currentTrainingData.Merge(new DataAccess(this, data).ToDataViewFiltered(row => !invalidRows.Contains(row.Position)));
                  using var temp = new DataStorageBinaryTempFile();
                  SaveData(temp, merged);
                  SaveData(trainingStorage, LoadData(temp));
               }
               else
                  SaveData(trainingStorage, new DataAccess(this, data).ToDataViewFiltered(row => !invalidRows.Contains(row.Position)));
            }
            cancellation.ThrowIfCancellationRequested();
         }
      }
      /// <summary>
      /// Stoppa il training ed annulla la validita' del modello
      /// </summary>
      public void ClearModel()
      {
         if (this.evaluator is Evaluator evaluator && evaluator != null) {
            // Stoppa il training
            StopTrainingInternal(evaluator);
            // Invalida la valutazione
            lock (evaluator)
               SetEvaluation(evaluator, null, null, evaluator.Timestamp);
            // Segnala variazione modello
            OnModelChanged(new ModelTrainingEventArgs(evaluator));
         }
      }
      /// <summary>
      /// Verifica la consistenza di due schemi
      /// </summary>
      /// <param name="schema1">Primo schema</param>
      /// <param name="schema2">Secondo schema</param>
      /// <param name="errMsg">Eventuale messaggio di errore personalizzato</param>
      private void CheckSchemaConsistence(DataViewSchema schema1, DataViewSchema schema2, string errMsg = null)
      {
         // REVIEW: Allow schema isomorphism.
         errMsg ??= "Inconsistent schema: all source dataviews must have identical column names, sizes, and item types.";
         var colCount = schema1.Count;
         // Check if the column counts are identical.
         Channel.Check(colCount == schema2.Count, errMsg);
         for (int c = 0; c < colCount; c++) {
            Channel.Check(schema1[c].Name == schema2[c].Name, errMsg);
            Channel.Check(schema1[c].Type.SameSizeAndItemType(schema2[c].Type), errMsg);
         }
      }
      /// <summary>
      /// Commit dei dati di training
      /// </summary>
      /// <param name="evaluator">L'evaluator</param>
      private void CommitTrainingData(Evaluator evaluator)
      {
         // Verifica degli storage
         if (evaluator.DataStorage is not IDataStorage dataStorage || evaluator.TrainingStorage is not IDataStorage trainingStorage)
            return;
         var tmpFileName = Path.GetTempFileName();
         try {
            // Dati di storage e di training concatenati
            var mergedData = LoadData(dataStorage);
            mergedData = mergedData?.Merge(LoadData(trainingStorage)) ?? LoadData(trainingStorage);
            // Salva in un file temporaneo il merge
            var tmpStorage = new DataStorageBinaryFile(tmpFileName) { KeepHidden = true };
            SaveData(tmpStorage, mergedData);
            // Aggiorna lo storage
            SaveData(dataStorage, LoadData(tmpStorage));
            // Cancella i dati di training
            SaveData(trainingStorage, DataViewGrid.Create(this, mergedData.Schema));
            // Genera evento
            OnTrainingDataChanged(new ModelTrainingEventArgs(evaluator));

         }
         finally {
            try {
               // Cancella il file temporaneo
               FileUtil.Delete(tmpFileName);
            }
            catch (Exception) {
            }
         }
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di dispose da programma</param>
      protected override void Dispose(bool disposing)
      {
         base.Dispose(disposing);
         try {
            if (context is MachineLearningContext ml)
               ml.RemoveDisposable(this);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
         try {
            (evaluator?.Model as IDisposable)?.Dispose();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
         if (evaluator != null)
            evaluator.Model = null;
      }
      /// <summary>
      /// Funzione di restituzione della migliore fra due valutazioni modello
      /// </summary>
      /// <param name="modelEvaluation1">Prima valutazione</param>
      /// <param name="modelEvaluation2">Seconda valutazione</param>
      /// <returns>La migliore delle due valutazioni</returns>
      /// <remarks>Tenere conto che le valutazioni potrebbero essere null</remarks>
      protected virtual object GetBestModelEvaluation(object modelEvaluation1, object modelEvaluation2) => modelEvaluation1;
      /// <summary>
      /// Funzione di ottenimento del provider di canali
      /// </summary>
      /// <returns>Il provider</returns>
      protected override sealed IChannelProvider GetChannelProvider() => context;
      /// <summary>
      /// Get the evaluation
      /// </summary>
      /// <param name="cancellation">Optional cancellation token</param>
      /// <returns>La valutazione</returns>
      protected IModelEvaluator GetEvaluation(CancellationToken cancellation = default)
      {
         // Get the evaluator
         var evaluator = GetEvaluator(cancellation);
         // Wait the evaluation or the cancellation
         var waitResult = Task.Run(() => WaitHandle.WaitAny(new[] { evaluator.Available, cancellation.WaitHandle, evaluator.Cancellation.WaitHandle })).Result;
         // Propagate the exception if the task was not the evaluation task
         switch (waitResult) {
            case 0:
               break;
            case 1:
               throw new OperationCanceledException();
            case 2:
               // Await to propagate the possible exception
               throw evaluator.Task.Exception;
         }
         // Verify that a result is really available
         if (!evaluator.Available.WaitOne(0))
            throw new OperationCanceledException();
         return evaluator;
      }
      /// <summary>
      /// Restituisce l'evaluator
      /// </summary>
      /// <param name="trainer">Il trainer</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task</returns>
      protected IModelEvaluator GetEvaluator(CancellationToken cancellation = default)
      {
         var currentEvaluator = default(Evaluator);
         var stopEvaluator = default(Evaluator);
         var startEvaluator = default(Evaluator);
         lock (this) {
            // Avvia il task di training se necessario
            currentEvaluator = evaluator;
            var startCurrentEvaluator = false;
            var createEvaluator = false;
            var _dataStorage = (this as IDataStorageProvider)?.DataStorage ?? this as IDataStorage;
            var _modelStorage = (this as IModelStorageProvider)?.ModelStorage ?? this as IModelStorage;
            var _trainingStorage = (this as ITrainingStorageProvider)?.TrainingStorage;
            var _inputSchema = (this as IInputSchema)?.InputSchema;
            var _modelAutoCommit = (this as IModelAutoCommit)?.ModelAutoCommit ?? false;
            var _modelAutoSave = (this as IModelAutoSave)?.ModelAutoSave ?? false;
            var _trainer = (this as IModelTrainerProvider)?.ModelTrainer ?? this as IModelTrainer;
            if (currentEvaluator != null) {
               if (currentEvaluator.TaskTraining.Task.IsCompleted || currentEvaluator.TaskTraining.CancellationToken.IsCancellationRequested) {
                  if (!currentEvaluator.Available.WaitOne(0))
                     startCurrentEvaluator = true;
                  else if (currentEvaluator.Trainer is IModelTrainerCycling cycling && currentEvaluator.TrainsCount < cycling.MaxTrainingCycles)
                     startCurrentEvaluator = true;
               }
               if (currentEvaluator.Timestamp != default) {
                  if (((currentEvaluator.TrainingStorage as IDataTimestamp)?.DataTimestamp ?? default) > currentEvaluator.Timestamp)
                     createEvaluator = true;
                  else if (((currentEvaluator.DataStorage as IDataTimestamp)?.DataTimestamp ?? default) > currentEvaluator.Timestamp)
                     createEvaluator = true;
                  else if (((currentEvaluator.ModelStorage as IDataTimestamp)?.DataTimestamp ?? default) > currentEvaluator.Timestamp)
                     createEvaluator = true;
               }
               else if (currentEvaluator.DataStorage != _dataStorage)
                  createEvaluator = true;
               else if (currentEvaluator.ModelStorage != _modelStorage)
                  createEvaluator = true;
               else if (currentEvaluator.TrainingStorage != _trainingStorage)
                  createEvaluator = true;
               else if (currentEvaluator.InputSchema != _inputSchema)
                  createEvaluator = true;
               else if (currentEvaluator.ModelAutoCommit != _modelAutoCommit)
                  createEvaluator = true;
               else if (currentEvaluator.ModelAutoSave != _modelAutoSave)
                  createEvaluator = true;
               else if (currentEvaluator.Trainer != _trainer)
                  createEvaluator = true;
            }
            else
               createEvaluator = true;
            if (createEvaluator)
               stopEvaluator = currentEvaluator;
            else if (currentEvaluator != null) {
               if (currentEvaluator.TaskTraining.Task.IsCompleted || currentEvaluator.TaskTraining.CancellationToken.IsCancellationRequested)
                  stopEvaluator = currentEvaluator;
            }
            if (createEvaluator) {
               (currentEvaluator?.Model as IDisposable)?.Dispose();
               startEvaluator = currentEvaluator = evaluator = new Evaluator
               {
                  DataStorage = _dataStorage,
                  InputSchema = _inputSchema,
                  Model = null,
                  ModelAutoCommit = _modelAutoCommit,
                  ModelAutoSave = _modelAutoSave,
                  ModelStorage = _modelStorage,
                  Timestamp = default,
                  Trainer = _trainer,
                  TrainingStorage = _trainingStorage,
                  TrainsCount = 0,
               };
            }
            else if (startCurrentEvaluator)
               startEvaluator = currentEvaluator;
         }
         if (stopEvaluator != null) {
            StopTrainingInternal(stopEvaluator);
            cancellation.ThrowIfCancellationRequested();
         }
         if (startEvaluator != null && startEvaluator.TaskTraining.Task.IsCompleted)
            StartTrainingInternal(startEvaluator, stopEvaluator != null ? stopEvaluator.TaskTraining.ParentCancellationToken : cancellation);
         cancellation.ThrowIfCancellationRequested();
         return currentEvaluator;
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected virtual object GetModelEvaluation(IDataTransformer model, IDataAccess data) => null;
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected virtual string GetModelEvaluationInfo(object modelEvaluation) => null;
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Riga di dati da usare per la previsione</param>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La previsione</returns>
      public IDataAccess GetPredictionData(IEnumerable<object> data, CancellationToken cancellation = default)
      {
         // Attande il modello od un eventuale errore di training
         var evaluation = GetEvaluation(cancellation);
         lock (evaluation) {
            cancellation.ThrowIfCancellationRequested();
            // Crea la vista di dati per la previsione
            var dataViewGrid = DataViewGrid.Create(this, evaluation.InputSchema);
            dataViewGrid.Add(data.ToArray());
            cancellation.ThrowIfCancellationRequested();
            // Effettua la predizione
            var prediction = new DataAccess(this, evaluation.Model.Transform(dataViewGrid, cancellation));
            cancellation.ThrowIfCancellationRequested();
            return prediction;
         }
      }
      /// <summary>
      /// Restituisce il modello sottoposto al training
      /// </summary>
      /// <param name="trainer">Il trainer da utilizzare</param>
      /// <param name="data">Dati di training</param>
      /// <param name="metrics">Eventuale metrica</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il transformer di dati</returns>
      protected virtual IDataTransformer GetTrainedModel(IModelTrainer trainer, IDataAccess data, out object metrics, CancellationToken cancellation)
      {
         metrics = null;
         return null;
      }
      /// <summary>
      /// Carica i dati da uno storage
      /// </summary>
      /// <param name="dataStorage">Storage di dati</param>
      /// <returns>La vista di dati</returns>
      public abstract IDataAccess LoadData(IDataStorage dataStorage);
      /// <summary>
      /// Carica il modello da uno storage
      /// </summary>
      /// <param name="modelStorage">Storage del modello</param>
      /// <param name="schema">Lo schema del modello</param>
      /// <returns>Il modello</returns>
      public abstract IDataTransformer LoadModel(IModelStorage modelStorage, out DataViewSchema schema);
      /// <summary>
      /// Funzione di notifica della variazione del modello
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnModelChanged(ModelTrainingEventArgs e)
      {
         try {
            ModelChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di notifica della variazione dei dati di training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnTrainingDataChanged(ModelTrainingEventArgs e)
      {
         try {
            TrainingDataChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di inizio ciclo di training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnTrainingCycleStarted(ModelTrainingEventArgs e)
      {
         try {
            Channel.WriteLog($"Training cycle {e.Evaluator.TrainsCount}");
            TrainingCycleStarted?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di notifica della fine del training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnTrainingEnded(ModelTrainingEventArgs e)
      {
         try {
            Channel.WriteLog("Training ended");
            Channel.WriteLog("--------------");
            TrainingEnded?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di notifica dello start del training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnTrainingStarted(ModelTrainingEventArgs e)
      {
         try {
            Channel.WriteLog("--------------");
            Channel.WriteLog("Training started");
            TrainingStarted?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Salva i dati in uno storage
      /// </summary>
      /// <param name="dataStorage">Storage di dati</param>
      /// <param name="data">Dati</param>
      public abstract void SaveData(IDataStorage dataStorage, IDataAccess data);
      /// <summary>
      /// Salva il modello in uno storage
      /// </summary>
      /// <param name="modelStorage">Storage del modello</param>
      /// <param name="model">Modello</param>
      /// <param name="schema">Lo schema del modello</param>
      /// <returns>Il modello</returns>
      public abstract void SaveModel(IModelStorage modelStorage, IDataTransformer model, DataViewSchema schema);
      /// <summary>
      /// Imposta i dati di valutazione
      /// </summary>
      /// <param name="model">Il modello</param>
      /// <param name="schema">Lo schema</param>
      /// <param name="timestamp">L'istante</param>
      /// <remarks>Se la valutazione e' nulla annulla la validita' dei dati</remarks>
      private void SetEvaluation(Evaluator evaluator, IDataTransformer model, DataViewSchema schema, DateTime timestamp)
      {
         lock (evaluator) {
            // Annulla il modello
            if (model == null) {
               var modelCleared = evaluator.Model != null;
               if (modelCleared)
                  (evaluator.Model as IDisposable)?.Dispose();
               evaluator.Model = null;
               evaluator.InputSchema = schema;
               evaluator.Timestamp = default;
               evaluator.Available.Reset();
               if (modelCleared)
                  Channel.WriteLog("Model cleared");
            }
            // Imposta il modello
            else {
               var modelSetted = this.evaluator.Model != model;
               if (modelSetted)
                  (evaluator.Model as IDisposable)?.Dispose();
               evaluator.Model = model;
               evaluator.InputSchema = schema;
               evaluator.Timestamp = timestamp;
               evaluator.Available.Set();
               if (modelSetted)
                  Channel.WriteLog("Model setted");
            }
         }
      }
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione del training</param>
      public void StartTraining(CancellationToken cancellation = default) => GetEvaluator(cancellation);
      /// <summary>
      /// FUnzione interna di start del training
      /// </summary>
      /// <param name="evaluator">L'evaluator</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task</returns>
      private void StartTrainingInternal(Evaluator evaluator, CancellationToken cancellation)
      {
         if (evaluator.TaskTraining.Task.IsCompleted) {
            Channel.WriteLog("Start training");
            evaluator.Available.Reset();
            evaluator.TaskTraining.StartNew(
               c =>
               Task.Factory.StartNew(
                  () =>
                  {
                     var task = default(Task);
                     using var cts = CancellationTokenSource.CreateLinkedTokenSource(c);
                     try {
                        task = TrainingAsync(evaluator, cts.Token);
                        if (context is MachineLearningContext ml)
                           ml.AddWorkingTask(task, cts);
                        task.Wait();
                     }
                     finally {
                        if (task != null && context is MachineLearningContext ml)
                           ml.RemoveWorkingTask(task);
                     }
                  },
                  c,
                  TaskCreationOptions.LongRunning,
                  TaskScheduler.Default),
               cancellation);
         }
      }
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      public void StopTraining(CancellationToken cancellation = default)
      {
         if (this.evaluator is Evaluator evaluator && evaluator != null)
            StopTrainingInternal(evaluator);
      }
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="evaluator">L'evaluator da stoppare</param>
      private void StopTrainingInternal(Evaluator evaluator)
      {
         if (!evaluator.TaskTraining.CancellationToken.IsCancellationRequested) {
            evaluator.TaskTraining.Cancel();
            Channel.WriteLog("Stop training");
            try {
               evaluator.TaskTraining.Task.Wait();
            }
            catch (OperationCanceledException) {
            }
         }
      }
      /// <summary>
      /// Routine di training continuo
      /// </summary>
      /// <param name="e">L'evaluator</param>
      /// <param name="cancel">Token di cancellazione</param>
      private async Task TrainingAsync(Evaluator e, CancellationToken cancel)
      {
         // Token di cancellazione del trainer del modello
         using var linkedCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancel);
         // Task di salvataggio modello
         var taskSaveModel = Task.CompletedTask;
         try {
            // Funzione di caricamento dati di storage e training mergiati
            IDataAccess LoadData()
            {
               var storageData = e.DataStorage == null ? null : this.LoadData(e.DataStorage);
               if (storageData != null && e.InputSchema != null)
                  CheckSchemaConsistence(storageData.Schema, e.InputSchema, "Inconsistent schema: data storage and model must have identical column names, sizes, and item types.");
               var trainingData = e.TrainingStorage == null ? null : this.LoadData(e.TrainingStorage);
               if (trainingData != null && e.InputSchema != null)
                  CheckSchemaConsistence(trainingData.Schema, e.InputSchema, "Inconsistent schema: training storage and model must have identical column names, sizes, and item types.");
               if (storageData != null)
                  return trainingData != null ? storageData.Merge(trainingData) : storageData;
               return trainingData;
            }
            // Azzera contatore di train
            e.TrainsCount = 0;
            // Segnala lo start del training
            cancel.ThrowIfCancellationRequested();
            OnTrainingStarted(new ModelTrainingEventArgs(e));
            // Validita' modello attuale
            var validModel =
               e.Model != null &&
               e.Timestamp != default &&
               e.Timestamp >= ((e.DataStorage as IDataTimestamp)?.DataTimestamp ?? default) &&
               e.Timestamp >= ((e.TrainingStorage as IDataTimestamp)?.DataTimestamp ?? default);
            // Definizioni
            var data = default(IDataAccess);
            var eval1 = default(object);
            var eval2 = default(object);
            var firstRun = true;
            var inputSchema = e.InputSchema;
            var model = validModel ? e.Model : null;
            var timestamp = validModel ? e.Timestamp : default;
            // Loop di training continuo
            while (!cancel.IsCancellationRequested) {
               // Segnala lo start di un ciclo di training
               OnTrainingCycleStarted(new ModelTrainingEventArgs(e));
               // Primo giro
               if (firstRun) {
                  firstRun = false;
                  // Carica il modello
                  var loadExistingModel =
                     ((e.ModelStorage as IDataTimestamp)?.DataTimestamp ?? default) >= ((e.DataStorage as IDataTimestamp)?.DataTimestamp ?? default) &&
                     ((e.ModelStorage as IDataTimestamp)?.DataTimestamp ?? default) >= ((e.TrainingStorage as IDataTimestamp)?.DataTimestamp ?? default) &&
                     (((e.ModelStorage as IDataTimestamp)?.DataTimestamp ?? default) > e.Timestamp || e.Model == null);
                  if (loadExistingModel) {
                     try {
                        Channel.WriteLog("Loading the model");
                        inputSchema = null;
                        model = e.ModelStorage == null ? null : await Task.Run(() => LoadModel(e.ModelStorage, out inputSchema), cancel);
                        timestamp = model == null ? default : (e.ModelStorage as IDataTimestamp)?.DataTimestamp ?? DateTime.UtcNow;
                        if (e.InputSchema != null && inputSchema != null) {
                           try {
                              CheckSchemaConsistence(e.InputSchema, inputSchema, "Inconsistent schema: input schema and stored model schema are different. The model will be discarded.");
                           }
                           catch (Exception) {
                              throw;
                           }
                        }
                     }
                     catch (OperationCanceledException) { }
                     catch (Exception exc) {
                        Channel.WriteLog($"Error loading the model: {exc.Message}");
                        Trace.WriteLine(exc);
                        model = null;
                        inputSchema = null;
                        timestamp = default;
                     }
                  }
                  cancel.ThrowIfCancellationRequested();
                  if (!loadExistingModel && model == null)
                     Channel.WriteLog("No model loaded. Retrain all");
                  else if (model == null)
                     Channel.WriteLog("No valid model present");
                  else if (loadExistingModel && model != null)
                     Channel.WriteLog("Model loaded");
                  // Carica i dati
                  data = LoadData();
                  cancel.ThrowIfCancellationRequested();
                  // Imposta la valutazione
                  var prevModel = e.Model;
                  SetEvaluation(e, model, e.InputSchema ?? inputSchema ?? data?.Schema, timestamp);
                  if (prevModel != model)
                     OnModelChanged(new ModelTrainingEventArgs(e));
                  cancel.ThrowIfCancellationRequested();
                  // Verifica l'esistenza di dati
                  if (data == null)
                     return;
                  // Effettua eventuale commit automatico
                  cancel.ThrowIfCancellationRequested();
                  if (e.ModelAutoCommit && data != null && e.DataStorage != null && e.TrainingStorage != null && (this.LoadData(e.TrainingStorage)?.GetRowCursor(e.InputSchema).MoveNext() ?? false)) {
                     Channel.WriteLog("Committing the new data");
                     await Task.Run(() => CommitTrainingData(e), cancel);
                  }
                  // Log della valutazione del modello
                  cancel.ThrowIfCancellationRequested();
                  if (loadExistingModel && model != null) {
                     eval1 = GetModelEvaluation(model, data);
                     cancel.ThrowIfCancellationRequested();
                     var evalInfo = GetModelEvaluationInfo(eval1);
                     if (!string.IsNullOrEmpty(evalInfo))
                        Channel.WriteLog(evalInfo);
                  }
                  // Azzera il contatore di retraining
                  cancel.ThrowIfCancellationRequested();
                  e.TrainsCount = 0;
               }
               // Ricarica i dati
               else {
                  // Effettua eventuale commit automatico
                  if (e.ModelAutoCommit && data != null && e.DataStorage != null && e.TrainingStorage != null && (this.LoadData(e.TrainingStorage)?.GetRowCursor(e.InputSchema).MoveNext() ?? false)) {
                     try {
                        Channel.WriteLog("Committing the new data");
                        await Task.Run(() => CommitTrainingData(e), cancel);
                        cancel.ThrowIfCancellationRequested();
                     }
                     catch (OperationCanceledException) {
                        throw;
                     }
                     catch (Exception exc) {
                        Channel.WriteLog($"Error committing the data: {exc.Message}");
                        Trace.WriteLine(exc);
                     }
                  }
                  data = LoadData();
                  // Verifica l'esistenza di dati
                  if (data == null)
                     return;
               }
               // Timestamp attuale
               timestamp = DateTime.UtcNow;
               cancel.ThrowIfCancellationRequested();
               // Verifica se non e' necessario un altro training
               if (e.Trainer is IModelTrainerCycling cyclingTrainer) {
                  if (e.Available.WaitOne(0)) {
                     if (e.TrainsCount >= cyclingTrainer.MaxTrainingCycles)
                        return;
                  }
               }
               else if (e.Available.WaitOne(0))
                  return;
               // Effettua la valutazione del modello corrente
               var currentModel = model;
               var taskEvaluate1 = eval1 != null || currentModel == null ? Task.FromResult(eval1) : Task.Run(() => GetModelEvaluation(currentModel, data), cancel);
               // Effettua il training
               var taskTrain = e.Trainer == null ? Task.FromResult(null as IDataTransformer) : Task.Run(() => GetTrainedModel(e.Trainer, data, out eval2, linkedCancellation.Token));
               // Messaggio di training ritardato
               cancel.ThrowIfCancellationRequested();
               var taskTrainingMessage = Task.Run(async () =>
               {
                  await Task.WhenAny(Task.Delay(250, cancel), taskTrain).ConfigureAwait(false);
                  if (taskTrain.IsCompleted && taskTrain.Result == default)
                     return;
                  cancel.ThrowIfCancellationRequested();
                  Channel.WriteLog(model == default ? "Training the model" : "Trying to find a better model");
               }, cancel);
               // Ottiene il risultato del training
               cancel.ThrowIfCancellationRequested();
               if ((model = await taskTrain) == null)
                  return;
               // Incrementa contatore di traning
               e.TrainsCount++;
               // Attende output log
               await taskTrainingMessage;
               eval2 ??= await Task.Run(() => GetModelEvaluation(model, data));
               // Attende la valutazione del primo modello
               eval1 = await taskEvaluate1;
               // Verifica se c'e' un miglioramento; se affermativo aggiorna la valutazione
               if (GetBestModelEvaluation(eval1, eval2) == eval2 || e?.Model == null) {
                  // Emette il log
                  Channel.WriteLog("Found suitable model");
                  var evalInfo = GetModelEvaluationInfo(eval2);
                  if (!string.IsNullOrEmpty(evalInfo))
                     Channel.WriteLog(evalInfo);
                  cancel.ThrowIfCancellationRequested();
                  // Eventuale salvataggio automatico modello
                  if (model != default && e.ModelAutoSave) {
                     Channel.WriteLog("Saving the new model");
                     await taskSaveModel;
                     cancel.ThrowIfCancellationRequested();
                     taskSaveModel = Task.Run(() =>
                     {
                        if (e.ModelStorage != null) {
                           SaveModel(e.ModelStorage, model, e.InputSchema);
                           e.Timestamp = (e.ModelStorage as IDataTimestamp)?.DataTimestamp ?? DateTime.UtcNow;
                        }
                     }, CancellationToken.None);
                  }
                  eval1 = eval2;
                  // Aggiorna la valutazione
                  cancel.ThrowIfCancellationRequested();
                  var prevModel = e.Model;
                  SetEvaluation(e, model, e.InputSchema, timestamp);
                  if (prevModel != model)
                     OnModelChanged(new ModelTrainingEventArgs(e));
                  cancel.ThrowIfCancellationRequested();
                  // Azzera coontatore retraining
                  e.TrainsCount = 0;
               }
               else
                  Channel.WriteLog("The model is worst than the current one; discarded.");
               // Verifica se deve uscire dal training perche' sono cambiate le condizioni del modello 
               if (e.DataStorage != ((this as IDataStorageProvider)?.DataStorage ?? this as IDataStorage))
                  break;
               else if (e.ModelStorage != ((this as IModelStorageProvider)?.ModelStorage ?? this as IModelStorage))
                  break;
               else if (e.TrainingStorage != (this as ITrainingStorageProvider)?.TrainingStorage)
                  break;
               else if (e.InputSchema != (this as IInputSchema)?.InputSchema)
                  break;
               else if (e.ModelAutoCommit != ((this as IModelAutoCommit)?.ModelAutoCommit ?? false))
                  break;
               else if (e.ModelAutoSave != ((this as IModelAutoSave)?.ModelAutoSave ?? false))
                  break;
            }
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            try {
               Channel.WriteLog(exc.ToString());
            }
            catch (Exception) {
            }
            throw;
         }
         finally {
            // Forza cancellazione task del trainer
            linkedCancellation.Cancel();
            // Attende il termine dei task
            try { await taskSaveModel; } catch { }
            // Segnala la fine del training
            if (!cancel.IsCancellationRequested)
               try { OnTrainingEnded(new ModelTrainingEventArgs(e)); } catch { }
         }
      }
      /// <summary>
      /// Trasforma i dati di input per il modello
      /// </summary>
      /// <param name="data">Dati di input</param>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      /// <returns>I dati trasformati</returns>
      public IDataAccess Transform(IDataAccess data, CancellationToken cancellation = default) =>
         new DataAccess(this, GetEvaluation(cancellation).Model?.Transform(data, cancellation));
      #endregion
   }

   /// <summary>
   /// Dati di valutazione
   /// </summary>
   public partial class ModelBase // Evaluator
   {
      private class Evaluator : IDisposable, IModelEvaluator
      {
         #region Properties
         /// <summary>
         /// Valutazione disponibile
         /// </summary>
         public EventWaitHandle Available { get; set; } = new ManualResetEvent(false);
         /// <summary>
         /// Storage di dati
         /// </summary>
         public IDataStorage DataStorage { get; set; }
         /// <summary>
         /// Membro di interfaccia Available
         /// </summary>
         WaitHandle IModelEvaluator.Available => Available;
         /// <summary>
         /// Membro di interfaccia
         /// </summary>
         CancellationToken IModelEvaluator.Cancellation => TaskTraining.CancellationToken;
         /// <summary>
         /// Membro di interfaccia
         /// </summary>
         Task IModelEvaluator.Task => TaskTraining.Task;
         /// <summary>
         /// Schema di input
         /// </summary>
         public DataViewSchema InputSchema { get; set; }
         /// <summary>
         /// Modello
         /// </summary>
         public IDataTransformer Model { get; set; }
         /// <summary>
         /// Abilitazione al commit automatico dei dati di training
         /// </summary>
         public bool ModelAutoCommit { get; set; }
         /// <summary>
         /// Abilitazione al salvataggio automatico del modello
         /// </summary>
         public bool ModelAutoSave { get; set; }
         /// <summary>
         /// Storage di dati
         /// </summary>
         public IModelStorage ModelStorage { get; set; }
         /// <summary>
         /// Task di training
         /// </summary>
         internal CancellableTask TaskTraining { get; set; } = new CancellableTask();
         /// <summary>
         /// Data e ora dell'evaluator
         /// </summary>
         public DateTime Timestamp { get; set; }
         /// <summary>
         /// Tipo di trainer da utilizzare
         /// </summary>
         public IModelTrainer Trainer { get; set; }
         /// <summary>
         /// Storage di dati di training
         /// </summary>
         public IDataStorage TrainingStorage { get; set; }
         /// <summary>
         /// Contatore di training
         /// </summary>
         public int TrainsCount { get; set; }
         #endregion
         #region Methods
         /// <summary>
         /// Funzione di cancellazione del task di valutazione
         /// </summary>
         public void Cancel() => TaskTraining.Cancel();
         /// <summary>
         /// Dispose dell'oggetto
         /// </summary>
         public void Dispose()
         {
            if (Available != null) {
               Available.Dispose();
               Available = null;
            }
            GC.SuppressFinalize(this);
         }
         #endregion
      }
   }
}
