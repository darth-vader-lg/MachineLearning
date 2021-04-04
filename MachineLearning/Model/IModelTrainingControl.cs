using System.Threading;
using System.Threading.Tasks;

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
      Task StartTrainingAsync(CancellationToken cancellation = default);
      /// <summary>
      /// Stoppa il training del modello
      /// </summary>
      /// <param name="cancellation">Eventuale token di cancellazione dell'attesa</param>
      Task StopTrainingAsync();
      #endregion
   }
}
