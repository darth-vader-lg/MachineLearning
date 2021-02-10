using MachineLearning.Data;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per il trainer automatico dei modelli
   /// </summary>
   public interface IModelTrainingAuto
   {
      #region Methods
      /// <summary>
      /// Effettua il training con la ricerca automatica del miglior trainer
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="maxTimeInSeconds">Numero massimo di secondi di training</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni incrociate</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il modello migliore</returns>
      IDataTransformer AutoTraining(IDataAccess data, int maxTimeInSeconds, out object metrics, int numberOfFolds = 1, CancellationToken cancellation = default);
      #endregion
   }
}
