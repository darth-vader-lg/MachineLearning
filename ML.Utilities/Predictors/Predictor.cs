using Microsoft.ML;
using Microsoft.ML.Data;
using ML.Utilities.Data;
using ML.Utilities.Models;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace ML.Utilities.Predictors
{
   /// <summary>
   /// Classe base per i predittori
   /// </summary>
   [Serializable]
   public abstract partial class Predictor
   {
      #region Fields
      /// <summary>
      /// Valutazione
      /// </summary>
      [NonSerialized]
      private Evaluator _evaluation;
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
      /// Gestore storage dati principale
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Valutazione
      /// </summary>
      public Evaluator Evaluation { get => _evaluation ??= new Evaluator(); private set => _evaluation = value; }
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      public MachineLearningContext ML { get; }
      /// <summary>
      /// Gestore storage modello
      /// </summary>
      public IModelStorage ModelStorage{ get; set; }
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
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public Predictor() : this(default(int?)) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public Predictor(int? seed) => ML = new MachineLearningContext(seed);
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public Predictor(MachineLearningContext ml) => ML = ml;
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
            Evaluation = evaluation;
            if (evaluation.Model != default)
               TaskEvaluation.TrySetResult();
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
