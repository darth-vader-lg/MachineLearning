using System.Threading;

namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per i trasformatori di dati
   /// </summary>
   public interface IDataTransformer
   {
      #region Methods
      /// <summary>
      /// Trasforma i dati
      /// </summary>
      /// <param name="data">Dati in ingresso</param>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      /// <returns>I dati trasformati</returns>
      IDataAccess Transform(IDataAccess data, CancellationToken cancellation = default);
      #endregion
   }
}
