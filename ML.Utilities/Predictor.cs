using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Classe base per i predittori
   /// </summary>
   [Serializable]
   public abstract partial class Predictor<T> : IDeserializationCallback
   {
      #region Fields
      /// <summary>
      /// Scheduler di creazione dell'oggetto
      /// </summary>
      [NonSerialized]
      private TaskScheduler _creationTaskScheduler;
      /// <summary>
      /// Thread di creazione dell'oggetto
      /// </summary>
      [NonSerialized]
      private Thread _creationThread;
      /// <summary>
      /// Gestore storage dati
      /// </summary>
      private IDataStorage _dataStorage;
      /// <summary>
      /// Valutazione
      /// </summary>
      [NonSerialized]
      private Evaluator _evaluation;
      /// <summary>
      /// Valutazione disponibile
      /// </summary>
      [NonSerialized]
      private EventWaitHandle _evaluationAvailable;
      /// <summary>
      /// Gestore storage modello
      /// </summary>
      private IModelStorage _modelStorage;
      /// <summary>
      /// Task di commit dei dati di training dati
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskCommitData;
      /// <summary>
      /// Task di salvataggio dati
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskSaveData;
      /// <summary>
      /// Task di salvataggio modello
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskSaveModel;
      /// <summary>
      /// Task di training
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskTraining;
      /// <summary>
      /// Dati aggiuntivi di training
      /// </summary>
      private DataStorageTextMemory _trainingData;
      #endregion
      #region Properties
      /// <summary>
      /// Abilitazione al commit automatico dei dati di training nello storage
      /// </summary>
      public bool AutoCommitData { get; set; }
      /// <summary>
      /// Abilitazione al salvataggio automatico del modello ogni volta che viene aggiornato 
      /// </summary>
      public bool AutoSaveModel { get; set; }
      /// <summary>
      /// Gestore storage dati principale
      /// </summary>
      public IDataStorage DataStorage
      {
         get => _dataStorage;
         set
         {
            if (value != _dataStorage) {
               _dataStorage = value;
               OnDataStorageChanged(EventArgs.Empty);
            }
         }
      }
      /// <summary>
      /// Valutazione
      /// </summary>
      public Evaluator Evaluation { get => _evaluation ??= new Evaluator(); private set => _evaluation = value; }
      /// <summary>
      /// Valutazione disponibile
      /// </summary>
      public EventWaitHandle EvaluationAvailable => _evaluationAvailable ??= new EventWaitHandle(false, EventResetMode.ManualReset);
      /// <summary>
      /// Nome della colonna label di ingresso
      /// </summary>
      protected string LabelColumnName { get; set; } = "Label";
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      public MachineLearningContext ML { get; }
      /// <summary>
      /// Gestore storage modello
      /// </summary>
      public IModelStorage ModelStorage
      {
         get => _modelStorage;
         set {
            if (value != _modelStorage) {
               _modelStorage = value;
               OnModelStorageChanged(EventArgs.Empty);
            }
         }
      }
      /// <summary>
      /// Nome dell'oggetto
      /// </summary>
      public string Name { get; set; }
      /// <summary>
      /// Indica necessita' di postare un azione nel thread di creazione dal momento che ci si trova in un altro
      /// </summary>
      public bool PostRequired => Thread.CurrentThread != _creationThread && _creationTaskScheduler != null;
      /// <summary>
      /// Nome della colonna di previsione
      /// </summary>
      protected string PredictionColumnName { get; set; } = "PredictedLabel";
      /// <summary>
      /// Abilita il salvataggio del commento dello schema di ingresso dei dati nel file (efficace solo su file di testo)
      /// </summary>
      public bool SaveDataSchemaComment { get; set; }
      /// <summary>
      /// Task di commit dei dati di training
      /// </summary>
      private CancellableTask TaskCommitData => _taskCommitData ??= new CancellableTask();
      /// <summary>
      /// Task di salvataggio dati
      /// </summary>
      private CancellableTask TaskSaveData => _taskSaveData ??= new CancellableTask();
      /// <summary>
      /// Task di salvataggio modello
      /// </summary>
      private CancellableTask TaskSaveModel => _taskSaveModel ??= new CancellableTask();
      /// <summary>
      /// Task di training
      /// </summary>
      private CancellableTask TaskTraining => _taskTraining ??= new CancellableTask();
      /// <summary>
      /// Dati aggiuntivi di training
      /// </summary>
      protected DataStorageTextMemory TrainingData => _trainingData ??= new DataStorageTextMemory();
      #endregion
      #region Events
      /// <summary>
      /// Evento di variazione storage dati
      /// </summary>
      public event EventHandler DataStorageChanged;
      /// <summary>
      /// Evento di variazione modello
      /// </summary>
      public event EventHandler ModelChanged;
      /// <summary>
      /// Evento di variazione storage modello
      /// </summary>
      public event EventHandler ModelStorageChanged;
      /// <summary>
      /// Evento di variazione dati di training
      /// </summary>
      public event EventHandler TrainingDataChanged;
      /// <summary>
      /// Evento di segnalazione training terminato
      /// </summary>
      public event EventHandler TrainingEnded;
      /// <summary>
      /// Evento di segnalazione training avviato
      /// </summary>
      public event EventHandler TrainingStarted;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public Predictor() : this(default(int?)) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public Predictor(int? seed) : this(new MachineLearningContext(seed)) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public Predictor(MachineLearningContext ml)
      {
         // Memorizza il contesto di machine learning
         ML = ml;
         // Memorizza lo scheduler e il thread di creazione
         _creationThread = Thread.CurrentThread;
         if (SynchronizationContext.Current != null)
            _creationTaskScheduler = TaskScheduler.FromCurrentSynchronizationContext();
      }
      /// <summary>
      /// Aggiunge un elenco di dati di training
      /// </summary>
      /// <param name="data">Elenco di linee di dati da aggiungere</param>
      public void AddTrainingData(IEnumerable<string> data)
      {
         // Annulla il training
         CancelTrainingAsync().ConfigureAwait(false).GetAwaiter().GetResult();
         // Aggiunge l'elenco di dati in coda
         var sb = new StringBuilder();
         if (!string.IsNullOrEmpty(TrainingData.TextData)) {
            sb.Append(TrainingData.TextData);
            if (!TrainingData.TextData.EndsWith(Environment.NewLine))
               sb.Append(Environment.NewLine);
         }
         foreach (var line in data) {
            if (!string.IsNullOrWhiteSpace(line))
               sb.AppendLine(line);
         }
         TrainingData.TextData = sb.ToString();
         OnTrainingDataChanged(EventArgs.Empty);
      }
      /// <summary>
      /// Aggiunge un elenco di dati di training definiti a colonne
      /// </summary>
      /// <param name="data">Valori della linea da aggiungere</param>
      public void AddTrainingData(IEnumerable<string[]> data) => AddTrainingData(from line in data select FormatDataRow(line));
      /// <summary>
      /// Aggiunge un dato di training definito a colonne
      /// </summary>
      /// <param name="data">Valori della linea da aggiungere</param>
      public void AddTrainingData(params string[] data) => AddTrainingData(new[] { FormatDataRow(data) } as IEnumerable<string>);
      /// <summary>
      /// Aggiunge un dato di training
      /// </summary>
      /// <param name="data">Linea di dati da aggiungere</param>
      public void AddTrainingData(string data) => AddTrainingData(new[] { data } as IEnumerable<string>);
      /// <summary>
      /// Stoppa il training ed annulla la validita' dei dati
      /// </summary>
      protected async Task CancelTrainingAsync()
      {
         // Stoppa il training
         await StopTrainingAsync().ConfigureAwait(false);
         // Invalida la valutazione
         SetEvaluation(null);
      }
      /// <summary>
      /// Commit dei dati
      /// </summary>
      public void CommitData() => CommitDataAsync().ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Commit asincrono dei dati
      /// </summary>
      /// <returns>Il Task</returns>
      public async Task CommitDataAsync()
      {
         if (string.IsNullOrWhiteSpace(TrainingData.TextData))
            return;
         await CancelTrainingAsync();
         await TaskTraining;
         await CommitDataAsyncInternal();
      }
      /// <summary>
      /// Commit asincrono dei dati interno
      /// </summary>
      /// <returns>Il Task</returns>
      private Task CommitDataAsyncInternal()
      {
         lock (TaskCommitData) {
            TaskCommitData.Task.ConfigureAwait(false).GetAwaiter().GetResult();
            return TaskCommitData.StartNew(c => Task.Run(async () =>
            {
               var data = LoadData(DataStorage, TrainingData);
               await SaveDataAsyncInternal(DataStorage, data);
               TrainingData.TextData = null;
            }, c));
         }
      }
      /// <summary>
      /// Formatta una riga di dati di input da un elenco di dati di input
      /// </summary>
      /// <param name="data"></param>
      /// <returns></returns>
      public string FormatDataRow(params string[] data)
      {
         // Linea da passare al modello
         var inputLine = new StringBuilder();
         // Quotatura stringhe
         var quote = ((DataStorage as IDataTextOptionsProvider)?.TextOptions?.AllowQuoting ?? true) ? "\"" : "";
         // Separatore di colonne
         var separatorChar = (DataStorage as IDataTextOptionsProvider)?.TextOptions?.Separators?.FirstOrDefault() ?? ',';
         // Loop di costruzione della linea di dati
         var separator = "";
         foreach (var item in data) {
            var text = item ?? "";
            var quoting = quote.Length > 0 && text.TrimStart().StartsWith(quote) && text.TrimEnd().EndsWith(quote) ? "" : quote;
            inputLine.Append($"{separator}{quoting}{text}{quoting}");
            separator = new string(separatorChar, 1);
         }
         return inputLine.ToString();
      }
      /// <summary>
      /// Restituisce l'evaluator
      /// </summary>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La valutazione</returns>
      public Evaluator GetEvaluator(CancellationToken cancellation = default) => GetEvaluatorAsync(cancellation).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce l'evaluator
      /// </summary>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La valutazione</returns>
      public async Task<Evaluator> GetEvaluatorAsync(CancellationToken cancellation = default)
      {
         // Attende la valutazione o la cancellazione del training
         var waitResult = await Task.Run(() => WaitHandle.WaitAny(new[] { EvaluationAvailable, cancellation.WaitHandle, TaskTraining.CancellationToken.WaitHandle }));
         // Se il task non era quello di valutazione valida propaga l'eventuale eccezione del task di training
         switch (waitResult) {
            case 0:
               break;
            case 1:
               throw new OperationCanceledException();
            case 2:
               // Await per propagare eventuale eccezione
               await TaskTraining;
               break;
         }
         // Verifica che sia veramente disponibile un risultato
         if (!EvaluationAvailable.WaitOne(0))
            throw new OperationCanceledException();
         return Evaluation;
      }
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public T GetPrediction(CancellationToken cancellation = default, params string[] data) => GetPredictionAsync(cancellation, data).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Elenco di dati da usare per la previsione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public T GetPrediction(IEnumerable<string> data) => GetPredictionAsync(data, CancellationToken.None).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Dati per la previsione</param>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La previsione</returns>
      public T GetPrediction(string data, CancellationToken cancellation = default) => GetPredictionAsync(data, cancellation).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<T> GetPredictionAsync(IEnumerable<string> data, CancellationToken cancellation = default) => GetPredictionAsync(FormatDataRow(data.ToArray()), cancellation);
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<T> GetPredictionAsync(CancellationToken cancellation = default, params string[] data) => GetPredictionAsync(FormatDataRow(data), cancellation);
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Dati per la previsione</param>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La previsione</returns>
      public async Task<T> GetPredictionAsync(string data, CancellationToken cancellation = default)
      {
         // Avvia il task di training
         _ = StartTrainingAsync(cancellation);
         // Crea una dataview con i dati di input
         var dataView = LoadData(new DataStorageTextMemory() { TextData = data, TextOptions = (DataStorage as IDataTextOptionsProvider)?.TextOptions ?? new TextLoader.Options() });
         cancellation.ThrowIfCancellationRequested();
         // Attande il modello od un eventuale errore di training
         var evaluator = await GetEvaluatorAsync(cancellation);
         cancellation.ThrowIfCancellationRequested();
         // Effettua la predizione
         var prediction = evaluator.Model.Transform(dataView);
         cancellation.ThrowIfCancellationRequested();
         // Verifica se l'oggetto possiede un costruttore che accetta come parametro la previsione
         if (typeof(T).GetConstructor(new[] { prediction.GetType() }) != null)
            return (T)Activator.CreateInstance(typeof(T), prediction);
         // Se il tipo di previsione e' una semplice IDataView
         if (typeof(T).IsAssignableFrom(typeof(IDataView)))
            return (T)prediction;
         // Se il tipo di previsione e' una semplice stringa
         else if (typeof(T).IsAssignableFrom(typeof(string)))
            return (T)(object)prediction.GetString(PredictionColumnName);
         // Previsione non ricostruibile
         return default;
      }
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      /// <param name="extra">Sorgenti extra</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(IDataStorage dataStorage = default, params IMultiStreamSource[] extra) => (dataStorage ?? DataStorage).LoadData(ML, extra);
      /// <summary>
      /// Carica il modello
      /// </summary>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      public ITransformer LoadModel(IModelStorage modelStorage = default) => (modelStorage ?? ModelStorage)?.LoadModel(ML, out _);
      /// <summary>
      /// Carica il modello
      /// </summary>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      public ITransformer LoadModel(out DataViewSchema inputSchema, IModelStorage modelStorage = default)
      {
         inputSchema = null;
         return (modelStorage ?? ModelStorage)?.LoadModel(ML, out inputSchema);
      }
      /// <summary>
      /// Funzione di notifica variazione storage dei dati
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnDataStorageChanged(EventArgs e)
      {
         try {
            CancelTrainingAsync().ConfigureAwait(false).GetAwaiter().GetResult();
            DataStorageChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione chiamata al termine della deserializzazione
      /// </summary>
      /// <param name="sender"></param>
      public virtual void OnDeserialization(object sender)
      {
         // Memorizza lo scheduler di creazione
         _creationThread = Thread.CurrentThread;
         if (SynchronizationContext.Current != null)
            _creationTaskScheduler = TaskScheduler.FromCurrentSynchronizationContext();
      }
      /// <summary>
      /// Funzione di notifica della variazione del modello
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnModelChanged(EventArgs e)
      {
         try {
            if (Evaluation.Model != default)
               ML.NET.WriteLog("Model setted", Name);
            ModelChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di notifica variazione storage del modello
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnModelStorageChanged(EventArgs e)
      {
         try {
            CancelTrainingAsync().ConfigureAwait(false).GetAwaiter().GetResult();
            ModelStorageChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di notifica della variazione dei dati di training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnTrainingDataChanged(EventArgs e)
      {
         try {
            CancelTrainingAsync().ConfigureAwait(false).GetAwaiter().GetResult();
            TrainingDataChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Funzione di notifica della fine del training
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnTrainingEnded(EventArgs e)
      {
         try {
            ML.NET.WriteLog("Training ended", Name);
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
      protected virtual void OnTrainingStarted(EventArgs e)
      {
         try {
            ML.NET.WriteLog("Training started", Name);
            TrainingStarted?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Posta un azione nel thread di creazione oggetto
      /// </summary>
      /// <param name="Action">Azione</param>
      public void Post(Action Action)
      {
         if (PostRequired)
            new Task(Action).Start(_creationTaskScheduler);
         else
            Action();
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      /// <param name="dataView">Dati</param>
      /// <param name="extra">Sorgenti extra di dati da accodare</param>
      public void SaveData(IDataStorage dataStorage = default, IDataView dataView = default, params IMultiStreamSource[] extra) => SaveDataAsync(dataStorage, dataView, extra).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Funzione di salvataggio asincrono dei dati
      /// </summary>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      /// <param name="dataView">Dati</param>
      /// <param name="extra">Sorgenti extra di dati da accodare</param>
      public async Task SaveDataAsync(IDataStorage dataStorage = default, IDataView dataView = default, params IMultiStreamSource[] extra)
      {
         await StopTrainingAsync();
         await TaskTraining;
         await SaveDataAsyncInternal(dataStorage, dataView, extra);
      }
      /// <summary>
      /// Funzione di salvataggio asincrono dei dati interna
      /// </summary>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      /// <param name="dataView">Dati</param>
      /// <param name="extra">Sorgenti extra di dati da accodare</param>
      private Task SaveDataAsyncInternal(IDataStorage dataStorage = default, IDataView dataView = default, params IMultiStreamSource[] extra)
      {
         lock (TaskSaveData) {
            TaskSaveData.Task.ConfigureAwait(false).GetAwaiter().GetResult();
            if ((dataStorage ??= DataStorage) == default || (dataView ??= Evaluation?.Data) == default)
               return Task.CompletedTask;
            return TaskSaveData.StartNew(c => Task.Run(() => dataStorage.SaveData(ML, dataView, SaveDataSchemaComment, extra), c));
         }
      }
      /// <summary>
      /// Salva il modello
      /// </summary>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      /// <param name="model">Eventuale modello</param>
      /// <param name="schema">Eventuale schema dei dati</param>
      public void SaveModel(IModelStorage modelStorage = default, ITransformer model = default, DataViewSchema schema = default) => SaveModelAsync(modelStorage, model, schema).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Funzione di salvataggio asincrono del modello
      /// </summary>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      /// <param name="model">Eventuale modello</param>
      /// <param name="schema">Eventuale schema dei dati</param>
      public async Task SaveModelAsync(IModelStorage modelStorage = default, ITransformer model = default, DataViewSchema schema = default)
      {
         await StopTrainingAsync();
         await TaskTraining;
         await SaveModelAsyncInternal(modelStorage, model, schema);
      }
      /// <summary>
      /// Funzione di salvataggio asincrono del modello
      /// </summary>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      /// <param name="model">Eventuale modello</param>
      /// <param name="schema">Eventuale schema dei dati</param>
      protected Task SaveModelAsyncInternal(IModelStorage modelStorage = default, ITransformer model = default, DataViewSchema schema = default)
      {
         lock (TaskSaveModel) {
            TaskSaveModel.Task.ConfigureAwait(false).GetAwaiter().GetResult();
            if ((modelStorage ??= ModelStorage) == default || (model ??= Evaluation?.Model) == default)
               return Task.CompletedTask;
            return TaskSaveModel.StartNew(c => Task.Run(() => modelStorage.SaveModel(ML, model, schema ?? Evaluation?.InputSchema), c));

         }
      }
      /// <summary>
      /// Imposta i dati di valutazione
      /// </summary>
      /// <param name="evaluation">Dati di valutazione</param>
      /// <remarks>Se la valutazione e' nulla annulla la validita' dei dati</remarks>
      protected void SetEvaluation(Evaluator evaluation)
      {
         // Annulla il modello
         if (evaluation == default)
            EvaluationAvailable.Reset();
         // Imposta il modello
         else {
            // Imposta la nuova valutazione
            Evaluation = evaluation;
            if (evaluation.Model != default)
               EvaluationAvailable.Set();
            // Segnala la vartiazione del modello
            OnModelChanged(EventArgs.Empty);
         }
      }
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione del training</param>
      public virtual async Task StartTrainingAsync(CancellationToken cancellation = default)
      {
         if (TaskTraining.Task.IsCompleted) {
            EvaluationAvailable.Reset();
            await TaskTraining.StartNew(c => Task.Factory.StartNew(() => Training(c), c, TaskCreationOptions.LongRunning, TaskScheduler.Default), cancellation);
         }
         else
            await TaskTraining;
      }
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione dell'attesa</param>
      public virtual async Task StopTrainingAsync(CancellationToken cancellation = default)
      {
         TaskTraining.Cancel();
         await TaskTraining.Task.ConfigureAwait(false);
      }
      /// <summary>
      /// Routine di training continuo
      /// </summary>
      /// <param name="cancel">Token di cancellazione</param>
      protected void Training(CancellationToken cancel)
      {
         try {
            // Segnala lo start del training
            cancel.ThrowIfCancellationRequested();
            OnTrainingStarted(EventArgs.Empty);
            // Carica il modello
            var model = default(ITransformer);
            var inputSchema = default(DataViewSchema);
            var loadExistingModel = string.IsNullOrEmpty(TrainingData.TextData);
            try { model = !loadExistingModel ? default : LoadModel(out inputSchema); } catch (Exception) { }
            cancel.ThrowIfCancellationRequested();
            ML.NET.WriteLog(!loadExistingModel ? "No model loaded. Retrain all" : model == default ? "No valid model present" : "Model loaded", Name);
            // Imposta le opzioni di testo per i dati extra in modo che siano uguali a quelli dello storage principale
            TrainingData.TextOptions = (DataStorage as IDataTextOptionsProvider)?.TextOptions ?? TrainingData.TextOptions;
            // Carica i dati
            var data = LoadData(DataStorage, TrainingData);
            cancel.ThrowIfCancellationRequested();
            // Imposta la valutazione
            SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = data?.Schema ?? inputSchema });
            // Effettua eventuale commit automatico
            var taskCommit = Task.CompletedTask;
            if (data != default && !string.IsNullOrEmpty(TrainingData.TextData) && AutoCommitData) {
               ML.NET.WriteLog("Committing the new data", Name);
               taskCommit = CommitDataAsyncInternal();
            }
            cancel.ThrowIfCancellationRequested();
            // Trainer
            var trainer = ML.NET.MulticlassClassification.Trainers.SdcaNonCalibrated();
            // Pipe di trasformazione
            var pipe =
               ML.NET.Transforms.Conversion.MapValueToKey("Label").
               Append(ML.NET.Transforms.Text.FeaturizeText("FeaturizeText", new TextFeaturizingEstimator.Options(), (from c in Evaluation.InputSchema where c.Name != "Label" select c.Name).ToArray())).
               Append(ML.NET.Transforms.CopyColumns("Features", "FeaturizeText")).
               Append(ML.NET.Transforms.NormalizeMinMax("Features")).
               AppendCacheCheckpoint(ML.NET).
               Append(trainer).
               Append(ML.NET.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // Loop di training continuo
            taskCommit.ConfigureAwait(false).GetAwaiter().GetResult();
            cancel.ThrowIfCancellationRequested();
            var prevMetrics = model == default ? default : ML.NET.MulticlassClassification.Evaluate(model.Transform(data));
            var seed = 0;
            var iterationMax = 10;
            var iteration = iterationMax;
            while (!cancel.IsCancellationRequested && --iteration >= 0) {
               try {
                  // Effettua il training
                  ML.NET.WriteLog(model == default ? "Training the model" : "Trying to find a better model", Name);
                  var dataView = ML.NET.Data.ShuffleRows(data, seed++);
                  model = pipe.Fit(dataView);
                  var metrics = ML.NET.MulticlassClassification.Evaluate(model.Transform(dataView));
                  //var crossValidation = ml.MulticlassClassification.CrossValidate(dataView, pipe, 5, "Label", null, seed++);
                  //var best = crossValidation.Best();
                  //var model = best.Model;
                  //var metrics = best.Metrics;
                  cancel.ThrowIfCancellationRequested();
                  // Verifica se c'e' un miglioramento; se affermativo salva il nuovo modello
                  if (prevMetrics == default || Evaluation?.Model == default || (metrics.MicroAccuracy >= prevMetrics.MicroAccuracy && metrics.LogLoss < prevMetrics.LogLoss)) {
                     // Emette il log
                     ML.NET.WriteLog("Found suitable model", Name);
                     ML.NET.WriteLog(metrics.ToText(), Name);
                     ML.NET.WriteLog(metrics.ConfusionMatrix.GetFormattedConfusionTable(), Name);
                     cancel.ThrowIfCancellationRequested();
                     // Eventuale salvataggio automatico modello
                     if (model != default && AutoSaveModel) {
                        ML.NET.WriteLog("Saving the new model", Name);
                        SaveModelAsyncInternal(default, model, Evaluation.InputSchema);
                     }
                     prevMetrics = metrics;
                     // Aggiorna la valutazione
                     cancel.ThrowIfCancellationRequested();
                     SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = Evaluation.InputSchema });
                     iteration = iterationMax;
                  }
                  // Ricarica i dati
                  cancel.ThrowIfCancellationRequested();
                  // Effettua eventuale commit automatico
                  if (!string.IsNullOrEmpty(TrainingData.TextData) && AutoCommitData) {
                     try {
                        ML.NET.WriteLog("Committing the new data", Name);
                        CommitDataAsyncInternal().ConfigureAwait(false).GetAwaiter().GetResult(); ;
                        cancel.ThrowIfCancellationRequested();
                        data = LoadData(DataStorage);
                     }
                     catch (OperationCanceledException) {
                        throw;
                     }
                     catch (Exception exc) {
                        Debug.WriteLine(exc);
                        data = LoadData(DataStorage, TrainingData);
                     }
                  }
                  else
                     data = LoadData(DataStorage, TrainingData);
                  cancel.ThrowIfCancellationRequested();
               }
               catch (OperationCanceledException) { }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
                  throw;
               }
            }
         }
         catch (OperationCanceledException) { }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            throw;
         }
         finally {
            // Segnala la fine del training
            OnTrainingEnded(EventArgs.Empty);
         }
      }
      #endregion
   }

   /// <summary>
   /// Dati di valutazione
   /// </summary>
   public partial class Predictor<T> // Evaluator
   {
      [Serializable]
      public class Evaluator
      {
         #region Properties
         /// <summary>
         /// Dati del modello
         /// </summary>
         public IDataView Data { get; set; }
         /// <summary>
         /// Modello
         /// </summary>
         public ITransformer Model { get; set; }
         /// <summary>
         /// Schema di input
         /// </summary>
         public DataViewSchema InputSchema { get; set; }
         #endregion
      }
   }
}
