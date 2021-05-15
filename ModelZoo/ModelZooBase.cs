using MachineLearning.Model;
using System;
using System.Diagnostics;
using System.Threading;

namespace MachineLearning.ModelZoo
{
   /// <summary>
   /// Classe base per lo zoo dei modelli
   /// </summary>
   [Serializable]
   public abstract class ModelZooBase<TModel> : IDisposable where TModel: IDisposable, IModelTrainingControl
   {
      #region Fields
      /// <summary>
      /// Indicatore di oggetto disposto
      /// </summary>
      [NonSerialized]
      private bool disposedValue;
      #endregion
      #region Properties
      /// <summary>
      /// Modello
      /// </summary>
      protected internal TModel Model { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Implementazione della IDisposable
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
         if (!disposedValue) {
            try {
               Model?.Dispose();
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
            disposedValue = true;
         }
      }
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione del training</param>
      public void StartTraining(CancellationToken cancellation = default) => Model.StartTraining(cancellation);
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione dell'attesa</param>
      public void StopTraining(CancellationToken cancellation = default) => Model.StopTraining(cancellation);
      #endregion
   }
}
