using MachineLearning.Data;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per il trainer standard dei modelli
   /// </summary>
   public interface IModelTrainingStandard
   {
      #region Methods
      /// <summary>
      /// Restituisce il modello effettuando il training standard
      /// </summary>
      /// <param name="data">Dati di training</param>
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      IDataTransformer StandardTraining(IDataAccess data, out object evaluationMetrics, CancellationToken cancellation = default);
      #endregion
   }
}
