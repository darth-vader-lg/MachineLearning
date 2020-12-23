using Microsoft.ML;
using Microsoft.ML.Data;
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
   public abstract partial class Predictor : IDeserializationCallback
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
      /// Task di training
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskTraining;
      /// <summary>
      /// Opzioni di caricamento testi di default
      /// </summary>
      private TextLoaderOptions _textLoaderOptionsDefault;
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
      private IDataStorage DataStorage => (this as IDataStorageProvider)?.DataStorage ?? this as IDataStorage;
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
      private IModelStorage ModelStorage => (this as IModelStorageProvider)?.ModelStorage ?? this as IModelStorage;
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
      /// Task di training
      /// </summary>
      private CancellableTask TaskTraining => _taskTraining ??= new CancellableTask();
      /// <summary>
      /// Gestore storage dati principale
      /// </summary>
      private TextLoaderOptions TextLoaderOptions => (this as ITextOptionsProvider)?.TextOptions ?? (_textLoaderOptionsDefault ??= new TextLoaderOptions());
      /// <summary>
      /// Dati aggiuntivi di training
      /// </summary>
      protected DataStorageTextMemory TrainingData => _trainingData ??= new DataStorageTextMemory();
      #endregion
      #region Events
      /// <summary>
      /// Evento di variazione modello
      /// </summary>
      public event EventHandler ModelChanged;
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
      /// Cancella l'elenco di dati di training
      /// </summary>
      public void ClearTrainingData()
      {
         // Annulla il training
         CancelTrainingAsync().ConfigureAwait(false).GetAwaiter().GetResult();
         // Cancella i dati di training
         TrainingData.TextData = "";
         OnTrainingDataChanged(EventArgs.Empty);
      }
      /// <summary>
      /// Commit dei dati
      /// </summary>
      /// <returns>Il Task</returns>
      private void CommitData()
      {
         var opt = (this as ITextOptionsProvider)?.TextOptions;
         var data = DataStorage?.LoadData(ML, opt, TrainingData);
         DataStorage?.SaveData(ML, data, opt, SaveDataSchemaComment);
         TrainingData.TextData = null;
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
         var quote = TextLoaderOptions.AllowQuoting ? "\"" : "";
         // Separatore di colonne
         var separatorChar = TextLoaderOptions.Separators?.FirstOrDefault() ?? ',';
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
      /// Funzione di restituzione della migliore fra due valutazioni modello
      /// </summary>
      /// <param name="modelEvaluation1">Prima valutazione</param>
      /// <param name="modelEvaluation2">Seconda valutazione</param>
      /// <returns>La migliore delle due valutazioni</returns>
      /// <remarks>Tenere conto che le valutazioni potrebbero essere null</remarks>
      protected virtual object GetBestModelEvaluation(object modelEvaluation1, object modelEvaluation2) => modelEvaluation1;
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
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected virtual object GetModelEvaluation(ITransformer model, IDataView data) => null;
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected virtual string GetModelEvaluationInfo(object modelEvaluation) => null;
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Elenco di dati da usare per la previsione</param>
      /// <returns>La previsione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public T GetPrediction<T>(params string[] data) => GetPredictionAsync<T>(data).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Elenco di dati da usare per la previsione</param>
      /// <returns>La previsione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public T GetPrediction<T>(IEnumerable<string> data) => GetPredictionAsync<T>(data).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Dati per la previsione</param>
      /// <returns>La previsione</returns>
      public T GetPrediction<T>(string data) => GetPredictionAsync<T>(data).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce il task di previsione
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<T> GetPredictionAsync<T>(IEnumerable<string> data, CancellationToken cancellation = default) => GetPredictionAsync<T>(FormatDataRow(data.ToArray()), cancellation);
      /// <summary>
      /// Restituisce il task di previsione
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<T> GetPredictionAsync<T>(CancellationToken cancellation = default, params string[] data) => GetPredictionAsync<T>(FormatDataRow(data), cancellation);
      /// <summary>
      /// Restituisce il task di previsione
      /// </summary>
      /// <param name="data">Dati per la previsione</param>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La previsione</returns>
      public async Task<T> GetPredictionAsync<T>(string data, CancellationToken cancellation = default)
      {
         // Avvia il task di training se necessario
         if (TrainingData.Timestamp > Evaluation.Timestamp || (DataStorage as ITimestamp)?.Timestamp > Evaluation.Timestamp)
            _ = StartTrainingAsync(cancellation);
         // Crea una dataview con i dati di input
         var dataView = new DataStorageTextMemory() { TextData = data }.LoadData(ML, TextLoaderOptions);
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
      /// Restituisce il modello effettuando il training. Da implementare nelle classi derivate
      /// </summary>
      /// <param name="dataView">Dati di training</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      protected virtual ITransformer GetTrainedModel(IDataView dataView, CancellationToken cancellation) => default;
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
      /// Imposta i dati di valutazione
      /// </summary>
      /// <param name="evaluation">Dati di valutazione</param>
      /// <remarks>Se la valutazione e' nulla annulla la validita' dei dati</remarks>
      protected void SetEvaluation(Evaluator evaluation)
      {
         // Annulla il modello
         if (evaluation == default) {
            Evaluation.Timestamp = default;
            EvaluationAvailable.Reset();
         }
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
            await TaskTraining.StartNew(c => Task.Factory.StartNew(async () => await TrainingAsync(c).ConfigureAwait(false), c, TaskCreationOptions.None, TaskScheduler.Default), cancellation);
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
      protected async Task TrainingAsync(CancellationToken cancel)
      {
         // Task di salvataggio modello
         var taskSaveModel = Task.CompletedTask;
         try {
            // Segnala lo start del training
            cancel.ThrowIfCancellationRequested();
            OnTrainingStarted(EventArgs.Empty);
            // Definizioni
            var data = default(IDataView);
            var eval1 = default(object);
            var eval2 = default(object);
            var firstRun = true;
            var inputSchema = default(DataViewSchema);
            var model = default(ITransformer);
            var timestamp = default(DateTime);
            // Loop di training continuo
            while (!cancel.IsCancellationRequested) {
               try {
                  // Primo giro
                  if (firstRun) {
                     firstRun = false;
                     // Carica il modello
                     var loadExistingModel = (ModelStorage as ITimestamp)?.Timestamp >= (DataStorage as ITimestamp)?.Timestamp && (ModelStorage as ITimestamp)?.Timestamp >= TrainingData.Timestamp;
                     if (loadExistingModel) {
                        try {
                           timestamp = DateTime.UtcNow;
                           model = await Task.Run(() => ModelStorage?.LoadModel(ML, out inputSchema), cancel);
                        }
                        catch (Exception) {
                           timestamp = default;
                        }
                     }
                     cancel.ThrowIfCancellationRequested();
                     ML.NET.WriteLog(!loadExistingModel ? "No model loaded. Retrain all" : model == default ? "No valid model present" : "Model loaded", Name);
                     // Carica i dati
                     data = DataStorage != null ? DataStorage.LoadData(ML, TextLoaderOptions, TrainingData) : TrainingData.LoadData(ML, TextLoaderOptions);
                     cancel.ThrowIfCancellationRequested();
                     // Imposta la valutazione
                     SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = data?.Schema ?? inputSchema, Timestamp = timestamp });
                     // Effettua eventuale commit automatico
                     cancel.ThrowIfCancellationRequested();
                     if (data != default && !string.IsNullOrEmpty(TrainingData.TextData) && DataStorage != default && AutoCommitData) {
                        ML.NET.WriteLog("Committing the new data", Name);
                        await Task.Run(() => CommitData(), cancel);
                     }
                     // Valuta il modello attuale
                     cancel.ThrowIfCancellationRequested();
                     eval2 = model == default ? default : await Task.Run(() => GetModelEvaluation(model, data), cancel);
                     // Verifica se deve ricalcolare il modello
                     if (GetBestModelEvaluation(eval1, eval2) != eval2)
                        return;
                     eval1 = eval2;
                  }
                  // Ricarica i dati
                  else {
                     // Effettua eventuale commit automatico
                     if (!string.IsNullOrEmpty(TrainingData.TextData) && DataStorage != default && AutoCommitData) {
                        try {
                           ML.NET.WriteLog("Committing the new data", Name);
                           await Task.Run(() => CommitData(), cancel);
                           cancel.ThrowIfCancellationRequested();
                           data = DataStorage?.LoadData(ML);
                        }
                        catch (OperationCanceledException) {
                           throw;
                        }
                        catch (Exception exc) {
                           Debug.WriteLine(exc);
                           data = DataStorage?.LoadData(ML, TextLoaderOptions, TrainingData);
                        }
                     }
                     else
                        data = DataStorage != null ? DataStorage.LoadData(ML, TextLoaderOptions, TrainingData) : TrainingData.LoadData(ML, TextLoaderOptions);
                  }
                  // Timestamp attuale
                  timestamp = DateTime.UtcNow;
                  // Effettua il training
                  cancel.ThrowIfCancellationRequested();
                  var taskTrain = Task.Run(() => GetTrainedModel(data, cancel));
                  // Messaggio di training ritardato
                  cancel.ThrowIfCancellationRequested();
                  var taskTrainingMessage = Task.Run(async () =>
                  {
                     await Task.WhenAny(Task.Delay(250, cancel), taskTrain);
                     if (taskTrain.IsCompleted && taskTrain.Result == default)
                        return;
                     cancel.ThrowIfCancellationRequested();
                     ML.NET.WriteLog(model == default ? "Training the model" : "Trying to find a better model", Name);
                  }, cancel);
                  // Ottiene il risultato del training
                  cancel.ThrowIfCancellationRequested();
                  if ((model = await taskTrain) == default)
                     return;
                  // Attende output log
                  await taskTrainingMessage;
                  eval2 = await Task.Run(() => GetModelEvaluation(model, data));
                  // Verifica se c'e' un miglioramento; se affermativo aggiorna la valutazione
                  if (GetBestModelEvaluation(eval1, eval2) == eval2 || Evaluation?.Model == default) {
                     // Emette il log
                     ML.NET.WriteLog("Found suitable model", Name);
                     var evalInfo = GetModelEvaluationInfo(eval2);
                     if (!string.IsNullOrEmpty(evalInfo))
                        ML.NET.WriteLog(evalInfo, Name);
                     cancel.ThrowIfCancellationRequested();
                     // Eventuale salvataggio automatico modello
                     if (model != default && AutoSaveModel) {
                        ML.NET.WriteLog("Saving the new model", Name);
                        await taskSaveModel;
                        cancel.ThrowIfCancellationRequested();
                        taskSaveModel = Task.Run(() => ModelStorage?.SaveModel(ML, model, Evaluation.InputSchema), CancellationToken.None);
                     }
                     eval1 = eval2;
                     // Aggiorna la valutazione
                     cancel.ThrowIfCancellationRequested();
                     SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = Evaluation.InputSchema, Timestamp = timestamp });
                  }
               }
               catch (OperationCanceledException) { throw; }
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
            // Attende il termine dei task
            try { await taskSaveModel; } catch { }
            // Segnala la fine del training
            OnTrainingEnded(EventArgs.Empty);
         }
      }
      #endregion
   }

   /// <summary>
   /// Dati di valutazione
   /// </summary>
   public partial class Predictor // Evaluator
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
         /// Schema di input
         /// </summary>
         public DataViewSchema InputSchema { get; set; }
         /// <summary>
         /// Modello
         /// </summary>
         public ITransformer Model { get; set; }
         /// <summary>
         /// Data e ora dell'evaluator
         /// </summary>
         public DateTime Timestamp { get; set; }
         #endregion
      }
   }
}
