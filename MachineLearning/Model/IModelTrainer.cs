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
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      ITransformer GetTrainedModel(ModelBase<MLContext> model, IDataAccess data, out object evaluationMetrics, CancellationToken cancellation);
      #endregion
   }
}
