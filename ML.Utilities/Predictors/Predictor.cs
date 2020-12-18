using Microsoft.ML;
using Microsoft.ML.Data;
using ML.Utilities.Data;
using ML.Utilities.Models;
using System;
using System.Diagnostics;
using System.Runtime.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace ML.Utilities.Predictors
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
      /// Gestore storage dati
      /// </summary>
      private IDataStorage _dataStorage;
      /// <summary>
      /// Valutazione
      /// </summary>
      [NonSerialized]
      private Evaluator _evaluation;
      /// <summary>
      /// Gestore storage modello
      /// </summary>
      private IModelStorage _modelStorage;
      /// <summary>
      /// Task di valutazione modello
      /// </summary>
      [NonSerialized]
      private TaskCompletionSource _taskEvaluation;
      /// <summary>
      /// Task di salvataggio asincrono dati
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskSaveData;
      /// <summary>
      /// Task di salvataggio asincrono modello
      /// </summary>
      [NonSerialized]
      private CancellableTask _taskSaveModel;
      #endregion
      #region Properties
      /// <summary>
      /// Scheduler di creazione dell'oggetto
      /// </summary>
      protected TaskScheduler CreationTaskScheduler => _creationTaskScheduler;
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
      /// Indica necessita' di invoke dal momento che non ci si trova nel contesto di creazione
      /// </summary>
      public bool InvokeRequired => Thread.CurrentThread != _creationThread;
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
      /// Abilita il salvataggio del commento dello schema di ingresso dei dati nel file (efficace solo su file di testo)
      /// </summary>
      public bool SaveDataSchemaComment { get; set; }
      /// <summary>
      /// Task di salvataggio dati
      /// </summary>
      private TaskCompletionSource TaskEvaluation { get => _taskEvaluation ??= new TaskCompletionSource(); set => _taskEvaluation = value; }
      /// <summary>
      /// Task di salvataggio dati
      /// </summary>
      private CancellableTask TaskSaveData { get => _taskSaveData ??= new CancellableTask(); set => _taskSaveData = value; }
      /// <summary>
      /// Task di salvataggio modello
      /// </summary>
      private CancellableTask TaskSaveModel { get => _taskSaveModel ??= new CancellableTask(); set => _taskSaveModel = value; }
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
         _creationTaskScheduler = TaskScheduler.Default == TaskScheduler.Current ? TaskScheduler.Current : TaskScheduler.FromCurrentSynchronizationContext();
      }
      /// <summary>
      /// Restituisce un task di attesa della valutazione copleta
      /// </summary>
      /// <returns></returns>
      public async Task<Evaluator> GetEvaluationAsync()
      {
         // Attende la valutazione
         await TaskEvaluation.Task;
         // Restituisce la valutazione
         return Evaluation;
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
            DataStorageChanged?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Invoca un azione nel thread di creazione oggetto
      /// </summary>
      /// <param name="Action">Azione</param>
      protected void Invoke(Action Action)
      {
         if (InvokeRequired)
            new Task(Action).Start(CreationTaskScheduler);
         else
            Action();
      }
      /// <summary>
      /// Funzione chiamata al termine della deserializzazione
      /// </summary>
      /// <param name="sender"></param>
      public virtual void OnDeserialization(object sender)
      {
         // Memorizza lo scheduler di creazione
         _creationThread = Thread.CurrentThread;
         _creationTaskScheduler = TaskScheduler.Default == TaskScheduler.Current ? TaskScheduler.Current : TaskScheduler.FromCurrentSynchronizationContext();
      }
      /// <summary>
      /// Funzione di notifica della variazione del modello
      /// </summary>
      /// <param name="e">Argomenti dell'evento</param>
      protected virtual void OnModelChanged(EventArgs e)
      {
         try {
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
            ModelStorageChanged?.Invoke(this, e);
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
            TrainingStarted?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      /// <param name="dataView">Dati</param>
      /// <param name="extra">Sorgenti extra di dati da accodare</param>
      public void SaveData(IDataStorage dataStorage = default, IDataView dataView = default, params IMultiStreamSource[] extra)
      {
         lock (TaskSaveData)
            (dataStorage ?? DataStorage)?.SaveData(ML, dataView ?? Evaluation.Data, SaveDataSchemaComment, extra);
      }
      /// <summary>
      /// Funzione di salvataggio asincrono dei dati
      /// </summary>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      /// <param name="dataView">Dati</param>
      /// <param name="extra">Sorgenti extra di dati da accodare</param>
      protected async Task SaveDataAsync(IDataStorage dataStorage = default, IDataView dataView = default, params IMultiStreamSource[] extra)
      {
         TaskSaveData.Cancel();
         await TaskSaveData;
         await TaskSaveData.StartNew(cancel => Task.Run(() =>
         {
            cancel.ThrowIfCancellationRequested();
            lock (TaskSaveData)
               SaveData(dataStorage, dataView, extra);
         }, cancel));
      }
      /// <summary>
      /// Salva il modello
      /// </summary>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      /// <param name="model">Eventuale modello</param>
      /// <param name="schema">Eventuale schema dei dati</param>
      public void SaveModel(IModelStorage modelStorage = default, ITransformer model = default, DataViewSchema schema = default)
      {
         lock (TaskSaveModel)
            (modelStorage ?? ModelStorage)?.SaveModel(ML, model ?? Evaluation.Model, schema ?? Evaluation.InputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio asincrono del modello
      /// </summary>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      /// <param name="model">Eventuale modello</param>
      /// <param name="schema">Eventuale schema dei dati</param>
      protected async Task SaveModelAsync(IModelStorage modelStorage = default, ITransformer model = default, DataViewSchema schema = default)
      {
         TaskSaveModel.Cancel();
         await TaskSaveModel;
         await TaskSaveModel.StartNew(cancel => Task.Run(() =>
         {
            cancel.ThrowIfCancellationRequested();
            lock (TaskSaveData)
               SaveModel(modelStorage, model, schema);
         }, cancel));
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
            TaskEvaluation.TrySetCanceled();
            TaskEvaluation = new TaskCompletionSource();
         }
         // Imposta il modello
         else {
            // Imposta la nuova valutazione
            Evaluation = evaluation;
            if (evaluation.Model != default)
               TaskEvaluation.TrySetResult();
            // Segnala la vartiazione del modello
            OnModelChanged(EventArgs.Empty);
         }
      }
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      public virtual async Task StartTrainingAsync(CancellationToken cancellation = default) => await Task.CompletedTask;
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      public virtual async Task StopTrainingAsync(CancellationToken cancellation = default) => await Task.CompletedTask;
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
