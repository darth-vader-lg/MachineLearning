using MachineLearning.Data;
using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i predittori
   /// </summary>
   [Serializable]
   public abstract partial class ModelBase : IDeserializationCallback, IMachineLearningContextProvider
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
      private TextLoader.Options _textLoaderOptionsDefault;
      /// <summary>
      /// Contatore di training
      /// </summary>
      [NonSerialized]
      private int _trainsCount;
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
      protected Evaluator Evaluation { get => _evaluation ??= new Evaluator(); private set => _evaluation = value; }
      /// <summary>
      /// Valutazione disponibile
      /// </summary>
      public EventWaitHandle EvaluationAvailable => _evaluationAvailable ??= new EventWaitHandle(false, EventResetMode.ManualReset);
      /// <summary>
      /// Nome colonna label
      /// </summary>
      public string LabelColumnName { get; protected set; } = "Label";
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      public MachineLearningContext ML { get; }
      /// <summary>
      /// Gestore storage modello
      /// </summary>
      private IModelStorage ModelStorage => (this as IModelStorageProvider)?.ModelStorage ?? this as IModelStorage;
      /// <summary>
      /// Gestore trainer modello
      /// </summary>
      private IModelTrainer ModelTrainer => (this as IModelTrainerProvider)?.ModelTrainer ?? this as IModelTrainer;
      /// <summary>
      /// Nome dell'oggetto
      /// </summary>
      public string Name { get; set; }
      /// <summary>
      /// Indica necessita' di postare un azione nel thread di creazione dal momento che ci si trova in un altro
      /// </summary>
      public bool PostRequired => Thread.CurrentThread != _creationThread && _creationTaskScheduler != null;
      /// <summary>
      /// Task di training
      /// </summary>
      private CancellableTask TaskTraining => _taskTraining ??= new CancellableTask();
      /// <summary>
      /// Gestore storage dati principale
      /// </summary>
      private TextLoader.Options TextLoaderOptions => (this as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? (_textLoaderOptionsDefault ??= new TextLoader.Options());
      /// <summary>
      /// Dati aggiuntivi di training
      /// </summary>
      private IDataStorage TrainingData => (this as ITrainingDataProvider)?.TrainingData;
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
      public ModelBase() : this(default(int?)) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public ModelBase(int? seed) : this(new MachineLearningContext(seed)) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public ModelBase(MachineLearningContext ml)
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
      /// <param name="data">Linee di dati da aggiungere</param>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      public void AddTrainingData(string data, bool checkForDuplicates = false)
      {
         // Verifica esistenza dati di training
         if (TrainingData is not IMultiStreamSource trainingDataSource || TrainingData is not IDataStorage trainingData)
            throw new InvalidOperationException("The object doesn't have training data characteristics");
         var sb = new StringBuilder();
         var hash = new HashSet<int>();
         var formatter = new DataStorageTextMemory();
         // Genera le hash per ciascuna riga del sorgente se e' abilitato il controllo dei duplicati 
         if (checkForDuplicates) {
            void AddHashes(IMultiStreamSource source)
            {
               if (source == null)
                  return;
               for (var i = 0; i < source.Count; i++) {
                  using var reader = source.OpenTextReader(i);
                  for (var line = reader.ReadLine(); line != null; line = reader.ReadLine()) {
                     formatter.TextData = line;
                     formatter.SaveData(this, formatter.LoadData(this));
                     hash.Add(formatter.TextData.GetHashCode());
                  }
               }
            }
            AddHashes(DataStorage as IMultiStreamSource);
            AddHashes(TrainingData as IMultiStreamSource);
         }
         // Aggiunge i dati di training preesistenti
         for (var i = 0; i < trainingDataSource.Count; i++) {
            using var reader = trainingDataSource.OpenTextReader(i);
            for (var line = reader.ReadLine(); line != null; line = reader.ReadLine()) {
               if (checkForDuplicates) {
                  formatter.TextData = line;
                  formatter.SaveData(this, formatter.LoadData(this));
                  if (!hash.Contains(formatter.TextData.GetHashCode())) {
                     hash.Add(formatter.TextData.GetHashCode());
                     sb.AppendLine(line);
                  }
               }
               else
                  sb.AppendLine(line);
            }
         }
         // Aggiunge le linee di training
         using var dataReader = new StringReader(data);
         for (var line = dataReader.ReadLine(); line != null; line = dataReader.ReadLine()) {
            if (checkForDuplicates) {
               formatter.TextData = line;
               formatter.SaveData(this, formatter.LoadData(this));
               if (!hash.Contains(formatter.TextData.GetHashCode())) {
                  hash.Add(formatter.TextData.GetHashCode());
                  sb.AppendLine(line);
               }
            }
            else
               sb.AppendLine(line);
         }
         // Aggiorna i dati di training
         if (sb.Length > 0) {
            // Annulla il training
            ClearTrainingAsync().ConfigureAwait(false).GetAwaiter().GetResult();
            // Formatta in testo e salva
            formatter.TextData = sb.ToString();
            trainingData.SaveData(this, formatter.LoadData(this));
            OnTrainingDataChanged(EventArgs.Empty);
         }
      }
      /// <summary>
      /// Aggiunge un dato di training definito a colonne
      /// </summary>
      /// <param name="checkForDuplicates">Controllo dei duplicati</param>
      /// <param name="data">Valori della linea da aggiungere</param>
      public void AddTrainingData(bool checkForDuplicates, params string[] data) => AddTrainingData(FormatDataRow(data), checkForDuplicates);
      /// <summary>
      /// Effettua il training con la ricerca automatica del miglior trainer
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="maxTimeInSeconds">Numero massimo di secondi di training</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni incrociate</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il modello migliore</returns>
      public abstract ITransformer AutoTraining(
         IDataAccess data,
         int maxTimeInSeconds,
         out object metrics,
         int numberOfFolds = 1,
         CancellationToken cancellation = default);
      /// <summary>
      /// Stoppa il training ed annulla la validita' dei dati
      /// </summary>
      public void ClearTraining() => ClearTrainingAsync().ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Stoppa il training ed annulla la validita' dei dati
      /// </summary>
      /// <param name="cancellation">Token di cancellazione</param>
      public async Task ClearTrainingAsync(CancellationToken cancellation = default)
      {
         // Stoppa il training
         await StopTrainingAsync(cancellation);
         // Invalida la valutazione
         SetEvaluation(null);
      }
      /// <summary>
      /// Cancella l'elenco di dati di training e le informazioni di training
      /// </summary>
      public void ClearTrainingData() => ClearTrainingDataAsync().ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Cancella l'elenco di dati di training e le informazioni di training
      /// </summary>
      /// <param name="cancellation">Token di cancellazione</param>
      public async Task ClearTrainingDataAsync(CancellationToken cancellation = default)
      {
         // Verifica che i dati di training siano validi
         if (TrainingData is not IDataStorage trainingData)
            throw new InvalidOperationException("The object doesn't have training data characteristics");
         // Annulla il training
         await ClearTrainingAsync(cancellation);
         // Cancella i dati di training
         var emptyData = new DataStorageTextMemory();
         cancellation.ThrowIfCancellationRequested();
         trainingData.SaveData(this, emptyData.LoadData(this));
         OnTrainingDataChanged(EventArgs.Empty);
      }
      /// <summary>
      /// Commit dei dati di training
      /// </summary>
      /// <returns>Il Task</returns>
      private void CommitTrainingData()
      {
         // Verifica degli storage
         if (DataStorage is not IDataStorage dataStorage || TrainingData is not IDataStorage trainingData)
            return;
         var tmpFileName = Path.GetTempFileName();
         try {
            // Dati di storage e di training concatenati
            var mergedData = dataStorage.LoadData(this).Merge(trainingData.LoadData(this));
            // Salva in un file temporaneo il merge
            var tmpStorage = new DataStorageBinaryFile(tmpFileName) { KeepHidden = true };
            tmpStorage.SaveData(this, mergedData);
            // Aggiorna lo storage
            dataStorage.SaveData(this, tmpStorage.LoadData(this));
            // Cancella i dati di training
            trainingData.SaveData(this, new DataStorageTextMemory().LoadData(this));
            OnTrainingDataChanged(EventArgs.Empty);
         }
         finally {
            try {
               // Cancella il file temporaneo
               File.Delete(tmpFileName);
            }
            catch (Exception) {
            }
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
      public abstract ITransformer CrossValidateTraining(
         IDataAccess data,
         out object metrics,
         int numberOfFolds = 5,
         string samplingKeyColumnName = null,
         int? seed = null);
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
      protected Evaluator GetEvaluator(CancellationToken cancellation = default) => GetEvaluatorAsync(cancellation).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce l'evaluator
      /// </summary>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La valutazione</returns>
      protected async Task<Evaluator> GetEvaluatorAsync(CancellationToken cancellation = default)
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
      protected virtual object GetModelEvaluation(ITransformer model, IDataAccess data) => null;
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected virtual string GetModelEvaluationInfo(object modelEvaluation) => null;
      /// <summary>
      /// Restituisce le pipe di training del modello
      /// </summary>
      /// <returns>Le pipe</returns>
      public abstract ModelPipes GetPipes();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Elenco di dati da usare per la previsione</param>
      /// <returns>La previsione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public IDataAccess GetPredictionData(params string[] data) => GetPredictionDataAsync(data).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Elenco di dati da usare per la previsione</param>
      /// <returns>La previsione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public IDataAccess GetPredictionData(IEnumerable<string> data) => GetPredictionDataAsync(data).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce la previsione
      /// </summary>
      /// <param name="data">Dati per la previsione</param>
      /// <returns>La previsione</returns>
      public IDataAccess GetPredictionData(string data) => GetPredictionDataAsync(data).ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Restituisce il task di previsione
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<IDataAccess> GetPredictionDataAsync(IEnumerable<string> data, CancellationToken cancellation = default) => GetPredictionDataAsync(FormatDataRow(data.ToArray()), cancellation);
      /// <summary>
      /// Restituisce il task di previsione
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<IDataAccess> GetPredictionDataAsync(CancellationToken cancellation = default, params string[] data) => GetPredictionDataAsync(FormatDataRow(data), cancellation);
      /// <summary>
      /// Restituisce il task di previsione
      /// </summary>
      /// <param name="data">Dati per la previsione</param>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La previsione</returns>
      public async Task<IDataAccess> GetPredictionDataAsync(string data, CancellationToken cancellation = default)
      {
         // Avvia il task di training se necessario
         var startTrain = false;
         if (ModelTrainer is IModelTrainerCycling cycling && _trainsCount < cycling.MaxTrainingCycles)
            startTrain = true;
         else if (((TrainingData as IDataTimestamp)?.DataTimestamp ?? default) > Evaluation.Timestamp || ((DataStorage as IDataTimestamp)?.DataTimestamp ?? default) > Evaluation.Timestamp)
            startTrain = true;
         if (startTrain) {
            await StopTrainingAsync(cancellation);
            _ = StartTrainingAsync(cancellation);
         }
         // Crea una dataview con i dati di input
         var inputData = new DataStorageTextMemory() { TextData = data }.LoadData(this);
         cancellation.ThrowIfCancellationRequested();
         // Attande il modello od un eventuale errore di training
         var evaluator = await GetEvaluatorAsync(cancellation);
         cancellation.ThrowIfCancellationRequested();
         // Effettua la predizione
         var prediction = new DataAccess(this, evaluator.Model.Transform(inputData));
         cancellation.ThrowIfCancellationRequested();
         return prediction;
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
            if (Evaluation.Model != null)
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
            ML.NET.WriteLog("--------------", Name);
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
            if (evaluation.Model != null)
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
            await TaskTraining.StartNew(
               c => Task.Factory.StartNew(
                  async () => await TrainingAsync(CancellationTokenSource.CreateLinkedTokenSource(c)).ConfigureAwait(false),
                  c,
                  TaskCreationOptions.LongRunning,
                  TaskScheduler.Default),
               cancellation);
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
         try {
            await TaskTraining.Task.ConfigureAwait(false);
         }
         catch (OperationCanceledException) {
         }
      }
      /// <summary>
      /// Routine di training continuo
      /// </summary>
      /// <param name="cancel">Token di cancellazione</param>
      protected async Task TrainingAsync(CancellationTokenSource cancelSource)
      {
         // Token di cancellazione
         var cancel = cancelSource.Token;
         // Funzione di caricamento dati di storage e training mergiati
         IDataAccess LoadData()
         {
            if (DataStorage != null) {
               if (TrainingData != null)
                  return DataStorage.LoadData(this).Merge(TrainingData.LoadData(this));
               else
                  return DataStorage.LoadData(this);
            }
            else if (TrainingData != null)
               return TrainingData.LoadData(this);
            return null;
         }
         // Task di salvataggio modello
         var taskSaveModel = Task.CompletedTask;
         try {
            // Azzera contatore di train
            _trainsCount = 0;
            // Segnala lo start del training
            cancel.ThrowIfCancellationRequested();
            OnTrainingStarted(EventArgs.Empty);
            // Validita' modello attuale
            var validModel =
               (Evaluation?.Timestamp ?? default) >= ((DataStorage as IDataTimestamp)?.DataTimestamp ?? default) &&
               (Evaluation?.Timestamp ?? default) >= ((TrainingData as IDataTimestamp)?.DataTimestamp ?? default);
            // Definizioni
            var data = default(IDataAccess);
            var eval1 = default(object);
            var eval2 = default(object);
            var firstRun = true;
            var inputSchema = Evaluation?.InputSchema;
            var model = validModel ? Evaluation?.Model : null;
            var timestamp = validModel ? (Evaluation?.Timestamp ?? default) : default;
            // Loop di training continuo
            while (!cancel.IsCancellationRequested) {
               try {
                  // Primo giro
                  if (firstRun) {
                     firstRun = false;
                     // Carica il modello
                     var loadExistingModel =
                        ((ModelStorage as IDataTimestamp)?.DataTimestamp ?? default) >= ((DataStorage as IDataTimestamp)?.DataTimestamp ?? default) &&
                        ((ModelStorage as IDataTimestamp)?.DataTimestamp ?? default) >= ((TrainingData as IDataTimestamp)?.DataTimestamp ?? default) &&
                        ((ModelStorage as IDataTimestamp)?.DataTimestamp ?? default) >= (Evaluation?.Timestamp ?? default);
                     if (loadExistingModel) {
                        try {
                           timestamp = DateTime.UtcNow;
                           ML.NET.WriteLog("Loading the model", Name);
                           model = await Task.Run(() => ModelStorage?.LoadModel(this, out inputSchema), cancel);
                        }
                        catch (Exception exc) {
                           ML.NET.WriteLog($"Error loading the model: {exc.Message}", Name);
                           timestamp = default;
                        }
                     }
                     cancel.ThrowIfCancellationRequested();
                     ML.NET.WriteLog(!loadExistingModel && model == null ? "No model loaded. Retrain all" : model == default ? "No valid model present" : "Model loaded", Name);
                     // Carica i dati
                     data = LoadData();
                     cancel.ThrowIfCancellationRequested();
                     // Imposta la valutazione
                     SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = data?.Schema ?? inputSchema, Timestamp = timestamp });
                     // Verifica l'esistenza di dati
                     if (data == null)
                        return;
                     // Effettua eventuale commit automatico
                     cancel.ThrowIfCancellationRequested();
                     if (data != null && DataStorage != null && TrainingData != null && AutoCommitData && TrainingData.LoadData(this).GetRowCursor(Evaluation.InputSchema).MoveNext()) {
                        ML.NET.WriteLog("Committing the new data", Name);
                        await Task.Run(() => CommitTrainingData(), cancel);
                     }
                     // Log della valutazione del modello
                     cancel.ThrowIfCancellationRequested();
                     if (loadExistingModel && model != null) {
                        eval1 = GetModelEvaluation(model, data);
                        cancel.ThrowIfCancellationRequested();
                        var evalInfo = GetModelEvaluationInfo(eval1);
                        if (!string.IsNullOrEmpty(evalInfo))
                           ML.NET.WriteLog(evalInfo, Name);
                     }
                     // Azzera il contatore di retraining
                     cancel.ThrowIfCancellationRequested();
                     _trainsCount = 0;
                  }
                  // Ricarica i dati
                  else {
                     // Effettua eventuale commit automatico
                     if (DataStorage != null && TrainingData != null && AutoCommitData && TrainingData.LoadData(this).GetRowCursor(Evaluation.InputSchema).MoveNext()) {
                        try {
                           ML.NET.WriteLog("Committing the new data", Name);
                           await Task.Run(() => CommitTrainingData(), cancel);
                           cancel.ThrowIfCancellationRequested();
                        }
                        catch (OperationCanceledException) {
                           throw;
                        }
                        catch (Exception exc) {
                           Debug.WriteLine(exc);
                        }
                     }
                     data = LoadData();
                     // Verifica l'esistenza di dati
                     if (data == default)
                        return;
                  }
                  // Timestamp attuale
                  timestamp = DateTime.UtcNow;
                  cancel.ThrowIfCancellationRequested();
                  // Verifica se non e' necessario un altro training
                  if (ModelTrainer is IModelTrainerCycling cyclingTrainer) {
                     if (EvaluationAvailable.WaitOne(0)) {
                        if (_trainsCount >= cyclingTrainer.MaxTrainingCycles)
                           return;
                     }
                  }
                  else if (EvaluationAvailable.WaitOne(0))
                     return;
                  // Effettua la valutazione del modello corrente
                  var currentModel = model;
                  var taskEvaluate1 = eval1 != null || currentModel == null ? Task.FromResult(eval1) : Task.Run(() => GetModelEvaluation(currentModel, data), cancel);
                  // Effettua il training
                  var taskTrain = Task.Run(() => ModelTrainer?.GetTrainedModel(this, data, out eval2, cancel));
                  // Messaggio di training ritardato
                  cancel.ThrowIfCancellationRequested();
                  var taskTrainingMessage = Task.Run(async () =>
                  {
                     await Task.WhenAny(Task.Delay(250, cancel), taskTrain).ConfigureAwait(false);
                     if (taskTrain.IsCompleted && taskTrain.Result == default)
                        return;
                     cancel.ThrowIfCancellationRequested();
                     ML.NET.WriteLog(model == default ? "Training the model" : "Trying to find a better model", Name);
                  }, cancel);
                  // Ottiene il risultato del training
                  cancel.ThrowIfCancellationRequested();
                  if ((model = await taskTrain) == default)
                     return;
                  // Incrementa contatore di traning
                  _trainsCount++;
                  // Attende output log
                  await taskTrainingMessage;
                  eval2 ??= await Task.Run(() => GetModelEvaluation(model, data));
                  // Attende la valutazione del primo modello
                  eval1 = await taskEvaluate1;
                  // Verifica se c'e' un miglioramento; se affermativo aggiorna la valutazione
                  if (GetBestModelEvaluation(eval1, eval2) == eval2 || Evaluation?.Model == null) {
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
                        taskSaveModel = Task.Run(() => ModelStorage?.SaveModel(this, model, Evaluation.InputSchema), CancellationToken.None);
                     }
                     eval1 = eval2;
                     // Aggiorna la valutazione
                     cancel.ThrowIfCancellationRequested();
                     SetEvaluation(new Evaluator { Data = data, Model = model, InputSchema = Evaluation.InputSchema, Timestamp = timestamp });
                     // Azzera coontatore retraining
                     _trainsCount = 0;
                  }
                  else
                     ML.NET.WriteLog("The model is worst than the current one; discarded.", Name);
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
            // Forza cancellazione per terminare altri task pendenti
            cancelSource.Cancel();
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
   public partial class ModelBase // Evaluator
   {
      [Serializable]
      protected class Evaluator
      {
         #region Properties
         /// <summary>
         /// Dati del modello
         /// </summary>
         public IDataAccess Data { get; set; }
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
