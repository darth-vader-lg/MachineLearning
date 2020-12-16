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
      /// Task di modello pronto
      /// </summary>
      private TaskCompletionSource<ModelAndSchema> taskModel = new TaskCompletionSource<ModelAndSchema>();
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
      /// Contesto ML
      /// </summary>
      public MLContext MLContext { get; }
      /// <summary>
      /// Modello
      /// </summary>
      public ITransformer Model { get { var t = taskModel.Task; return t.IsCompleted ? t.Result.Model : null; } }
      /// <summary>
      /// Schema di input del modello
      /// </summary>
      public DataViewSchema ModelInputSchema { get { var t = taskModel.Task; return t.IsCompleted ? t.Result.Schema : null; } }
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
      /// Carica i dati
      /// </summary>
      /// <param name="dataStorage">Eventuale gestore storage dati</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(IDataStorage dataStorage = null) => (dataStorage ?? DataStorage).LoadData(MLContext);
      /// <summary>
      /// Carica il modello
      /// </summary>
      /// <param name="modelStorage">Eventuale gestore storage modello</param>
      public void LoadModel(IModelStorage modelStorage = null)
      {
         var schema = default(DataViewSchema);
         SetModel((modelStorage ?? ModelStorage)?.LoadModel(MLContext, out schema), schema);
      }
      /// <summary>
      /// Salva il modello
      /// </summary>
      /// <param name="modelSaver">Eventuale oggetto di archiviazione modello</param>
      public void SaveModel(IModelStorage modelStorage = null)
      {
         lock (taskSaveModel.Locker)
            (modelStorage ?? ModelStorage)?.SaveModel(MLContext, Model, ModelInputSchema);
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
                  SaveModel();
               }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
                  throw;
               }
            }, cancellation.Token);
         }
      }
      /// <summary>
      /// Imposta il modello
      /// </summary>
      /// <param name="model">Modello</param>
      /// <param name="schema">Schema di input</param>
      protected void SetModel(ITransformer model, DataViewSchema schema = null)
      {
         // Annulla il modello
         if (model == default) {
            taskModel.TrySetCanceled();
            taskModel = new TaskCompletionSource<ModelAndSchema>();
         }
         // Imposta il modello
         else if (!taskModel.TrySetResult(new ModelAndSchema { Model = model, Schema = schema ?? ModelInputSchema })) {
            taskModel.Task.Result.Model = model;
            taskModel.Task.Result.Schema = schema ?? ModelInputSchema;
         }
      }
      #endregion
   }

   /// <summary>
   /// Modello e schema di input
   /// </summary>
   public partial class Predictor
   {
      [Serializable]
      private class ModelAndSchema
      {
         #region Properties
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
