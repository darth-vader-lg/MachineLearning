using Microsoft.ML;
using ML.Utilities.Data;
using ML.Utilities.Models;
using System;
using System.Diagnostics;
using System.Threading;
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
      /// Task di valutazione modello
      /// </summary>
      private TaskCompletionSource taskEvaluation = new TaskCompletionSource();
      /// <summary>
      /// Task di salvataggio asincrono modello
      /// </summary>
      private (Task Task, CancellationTokenSource Cancellation, object Locker) taskSaveModel = (Task.CompletedTask, new CancellationTokenSource(), new object());
      #endregion
      #region Properties
      /// <summary>
      /// Gestore storage dati
      /// </summary>
      public IDataStorage DataStorage { get; set; }
      /// <summary>
      /// Valutazione
      /// </summary>
      public Evaluator Evaluation { get; private set; }
      /// <summary>
      /// Contesto ML
      /// </summary>
      public MLContext MLContext { get; }
      /// <summary>
      /// Gestore storage modello
      /// </summary>
      public IModelStorage ModelStorage{ get; set; }
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
      public Predictor(int? seed) => MLContext = new MLContext(seed);
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public Predictor(MLContext ml) => MLContext = ml;
      /// <summary>
      /// Restituisce un task di attesa della valutazione copleta
      /// </summary>
      /// <returns></returns>
      public async Task<Evaluator> GetEvaluation()
      {
         // Attende la valutazione
         await taskEvaluation.Task;
         // Restituisce la valutazione
         return Evaluation;
      }
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(IDataStorage dataStorage = null) => (dataStorage ?? DataStorage).LoadData(MLContext);
      /// <summary>
      /// Carica il modello
      /// </summary>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      public ITransformer LoadModel(IModelStorage modelStorage = null) => (modelStorage ?? ModelStorage)?.LoadModel(MLContext, out _);
      /// <summary>
      /// Carica il modello
      /// </summary>
      /// <param name="schema">Schema di input del modello</param>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      public ITransformer LoadModel(out DataViewSchema schema, IModelStorage modelStorage = null)
      {
         schema = null;
         return (modelStorage ?? ModelStorage)?.LoadModel(MLContext, out schema);
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="dataStorage">Eventuale oggetto di archiviazione dati</param>
      public void SaveData(IDataStorage dataStorage = null)
      {
         lock (taskSaveModel.Locker)
            (dataStorage ?? DataStorage)?.SaveData(MLContext, Evaluation.Data);
      }
      /// <summary>
      /// Salva il modello
      /// </summary>
      /// <param name="model">Modello</param>
      /// <param name="modelStorage">Eventuale oggetto di archiviazione modello</param>
      public void SaveModel(IModelStorage modelStorage = null)
      {
         lock (taskSaveModel.Locker)
            (modelStorage ?? ModelStorage)?.SaveModel(MLContext, Evaluation.Model, Evaluation.Schema);
      }
      /// <summary>
      /// Funzione di salvataggio asincrono del modello
      /// </summary>
      protected void SaveModel()
      {
         lock (taskSaveModel.Locker) {
            taskSaveModel.Cancellation.Cancel();
            taskSaveModel.Task.Wait();
            var cancellation = taskSaveModel.Cancellation = new CancellationTokenSource();
            taskSaveModel.Task = Task.Run(() =>
            {
               try {
                  cancellation.Token.ThrowIfCancellationRequested();
                  SaveModel(ModelStorage);
               }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
                  throw;
               }
            }, cancellation.Token);
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
         if (evaluation == default) {
            taskEvaluation.TrySetCanceled();
            taskEvaluation = new TaskCompletionSource();
         }
         // Imposta il modello
         else {
            Evaluation = evaluation;
            if (evaluation.Model != default)
               taskEvaluation.TrySetResult();
         }
      }
      #endregion
   }

   /// <summary>
   /// Dati di valutazione
   /// </summary>
   public partial class Predictor
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
         public DataViewSchema Schema { get; set; }
         #endregion
      }
   }

}
