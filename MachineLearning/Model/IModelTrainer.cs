using MachineLearning.Data;
using Microsoft.ML;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i trainer dei modelli
   /// </summary>
   public interface IModelTrainer
   {
      #region Methods
      /// <summary>
      /// Restituisce il modello effettuando il training
      /// </summary>
      /// <param name="model">Modello con cui effettuare il training</param>
      /// <param name="data">Dati di training</param>
      /// <param name="pipe">Pipe del modello</param>
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      ITransformer GetTrainedModel(ModelBase model, IDataAccess data, IEstimator<ITransformer> pipe, out object evaluationMetrics, CancellationToken cancellation);
      #endregion
   }
}
