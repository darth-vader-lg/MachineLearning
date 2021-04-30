using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia di controllo dello stato del training
   /// </summary>
   public interface IModelTrainingControl
   {
      #region Methods
      /// <summary>
      /// Avvia il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione del training</param>
      void StartTraining(CancellationToken cancellation = default);
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione dell'attesa</param>
      void StopTraining(CancellationToken cancellation = default);
      #endregion
   }
}
