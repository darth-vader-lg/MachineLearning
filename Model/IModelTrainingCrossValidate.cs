using MachineLearning.Data;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i trainer di modelli con validazione incrociata
   /// </summary>
   public interface IModelTrainingCrossValidate
   {
      #region Methods
      /// <summary>
      /// Effettua il training con validazione incrociata del modello
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni</param>
      /// <param name="samplingKeyColumnName">Nome colonna di chiave di campionamento</param>
      /// <param name="seed">Seme per le operazioni random</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il modello migliore</returns>
      IDataTransformer CrossValidateTraining(IDataAccess data, out object metrics, int numberOfFolds = 5, string samplingKeyColumnName = null, int? seed = null, CancellationToken cancellation = default);
      #endregion
   }
}
