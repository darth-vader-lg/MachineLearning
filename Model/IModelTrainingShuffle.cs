using MachineLearning.Data;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per il trainer dei modelli con shuffle dei dati
   /// </summary>
   public interface IModelTrainingShuffle
   {
      #region Methods
      /// <summary>
      /// Restituisce il modello effettuando il training
      /// </summary>
      /// <param name="data">Dati di training</param>
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="seed">Seme per le operazioni random</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      IDataTransformer ShuffleTraining(IDataAccess data, out object evaluationMetrics, int? seed = null, CancellationToken cancellation = default);
      #endregion
   }
}
